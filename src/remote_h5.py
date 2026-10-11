"""A seekable read-only file over HTTP byte ranges, sized for feeding h5py.

Why this exists: `KOLF_Pan_Genome_QC_Filtered.h5ad` is 189 GB, the folder cap is 200 GB, and 93 GB
of it is already spent. The file cannot be stored, so it has to be streamed and thrown away, and
only the summary statistics kept -- which is the same contract every other ingester in `src/`
already honours.

Two MEASURED constraints shape this, neither obvious from the API:

  * figshare's `ndownloader` 302s to a presigned S3 URL carrying `X-Amz-Expires=10`. Ten seconds.
    So the signed URL is re-resolved whenever S3 answers 400/403, and a bare `curl -I` ALWAYS
    fails -- the HEAD lands after the window, which reads as "range reads are unsupported" when
    in fact S3 answers `206 Partial Content` quite happily.
  * throughput is LATENCY-bound, not bandwidth-bound: 3.2 MB/s at 1 MB blocks, 58.6 MB/s for one
    64 MB range, 202 MB/s with 8 concurrent 16 MB ranges. A 1 MB block turns a 90 GB pass into
    eight hours; 16 MB blocks with read-ahead turn it into minutes. Hence the pool.
"""
from __future__ import annotations

import http.client
import io
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class _Redirect(Exception):
    def __init__(self, url):
        self.url = url


class HTTPRangeFile(io.RawIOBase):
    """Seekable file over HTTP ranges. Prefetches ahead ONLY while access stays sequential.

    The sequential test is not decoration. h5py opens a file with a scatter of small metadata
    reads (superblock, B-trees, attributes); prefetching 8 x 16 MB behind each of those would
    fetch gigabytes to answer a few hundred bytes. Bulk reads of `data`/`indices` ARE sequential,
    which is where the whole cost lives, so the toggle costs nothing there and saves everything
    on the metadata pass.
    """

    def __init__(self, url, size, block=1 << 24, readahead=8):
        self.url, self.size, self.BLOCK = url, size, block
        self._pos, self._signed = 0, None
        self._cache, self._fut = {}, {}
        self._last_idx = -99
        self._pool = ThreadPoolExecutor(readahead, thread_name_prefix="range")
        self.readahead = readahead
        self.n_requests = self.n_bytes = 0

    # --- transport -------------------------------------------------------------------------
    def _resolve(self):
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise _Redirect(newurl)

        try:
            req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
            with urllib.request.build_opener(_NoRedirect).open(req, timeout=60):
                pass  # Close direct-host responses without downloading the source.
        except _Redirect as e:
            self._signed = e.url
            return self._signed
        # A host that serves the bytes directly (S3 public objects, e.g. clue.io's LINCS build)
        # rather than 302-ing to a presigned URL. That is not an error, it just means the URL
        # never expires -- so cache it and skip the re-resolve path in `_fetch` entirely.
        print(f"  [note] {self.url.rsplit('/', 1)[-1]}: served directly, no signed-URL redirect",
              file=sys.stderr)
        self._signed = self.url
        return self._signed

    def _fetch(self, start, end):
        """Inclusive byte range [start, end], refreshing the ~10 s signed URL on 400/403."""
        import random
        import time
        for attempt in range(8):
            try:
                # Resolution itself can receive GEO 503s and must use the same retries.
                url = self._signed or self._resolve()
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    # 200 means the server ignored Range and sent the whole object: only valid if that is the range
                    if resp.status != 206 and not (resp.status == 200 and start == 0 and end == self.size - 1):
                        raise RuntimeError(f"range read {start}-{end} returned HTTP {resp.status}")
                    status, content = resp.status, resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (400, 403):
                    print(f"  [warn] range {start}-{end} got HTTP {e.code}; "
                          f"refreshing URL, retry {attempt + 1}/8", file=sys.stderr)
                    self._signed = None          # expired presigned URL; re-resolve and retry
                    continue
                # NCBI's GEO ftp-over-https throttles a parallel range reader with 503s. They are
                # transient and per-request, so back off (exponentially, jittered) rather than lose
                # a multi-GB pass -- but say so, every time, so a throttled run is never silent.
                if e.code in (429, 500, 502, 503, 504):
                    w = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
                    print(f"  [warn] range {start}-{end} got HTTP {e.code}; retry "
                          f"{attempt + 1}/8 in {w:.1f}s", file=sys.stderr)
                    time.sleep(w)
                    self._signed = None
                    continue
                raise
            # IncompleteRead is the one that bit a 10 GB pass: S3 closes the connection mid-body and
            # `resp.read()` raises with a partial buffer. It is NOT a urllib.error, so without this
            # branch it escapes the retry loop and kills a multi-hour stream on one dropped socket.
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                    http.client.HTTPException, ConnectionError) as e:
                print(f"  [warn] range {start}-{end} failed ({type(e).__name__}: {e}); retrying",
                      file=sys.stderr)
                self._signed = None
                continue
            if status == 206 and len(content) != end - start + 1:
                print(f"  [warn] range {start}-{end} returned {len(content)} of "
                      f"{end - start + 1} bytes; retrying", file=sys.stderr)
                self._signed = None
                continue
            self.n_requests += 1
            self.n_bytes += len(content)
            return content
        raise RuntimeError(f"range read {start}-{end} failed after 8 attempts")

    # --- block cache -----------------------------------------------------------------------
    def _range_of(self, idx):
        start = idx * self.BLOCK
        return start, min(start + self.BLOCK, self.size) - 1

    def _block(self, idx):
        sequential = idx in (self._last_idx, self._last_idx + 1)
        self._last_idx = idx
        if idx not in self._cache and idx not in self._fut:
            self._fut[idx] = self._pool.submit(self._fetch, *self._range_of(idx))
        if sequential:
            for j in range(idx + 1, idx + 1 + self.readahead):
                if j * self.BLOCK < self.size and j not in self._cache and j not in self._fut:
                    self._fut[j] = self._pool.submit(self._fetch, *self._range_of(j))
        if idx not in self._cache:
            self._cache[idx] = self._fut.pop(idx).result()
        # Evict everything behind the cursor. Without this the cache IS the 189 GB file.
        for k in [k for k in self._cache if k < idx]:
            del self._cache[k]
        return self._cache[idx]

    # --- io.RawIOBase ----------------------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self._pos = {0: off, 1: self._pos + off, 2: self.size + off}[whence]
        return self._pos

    def tell(self):
        return self._pos

    def readinto(self, b):
        n = min(len(b), self.size - self._pos)
        if n <= 0:
            return 0
        out, pos = bytearray(), self._pos
        while len(out) < n:
            idx = pos // self.BLOCK
            chunk = self._block(idx)[pos - idx * self.BLOCK:]
            take = min(len(chunk), n - len(out))
            out += chunk[:take]
            pos += take
        b[:n] = out
        self._pos += n
        return n

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def figshare(file_id, size, **kw):
    """`HTTPRangeFile` over a figshare file id. Size must come from the API, not be guessed."""
    return HTTPRangeFile(f"https://ndownloader.figshare.com/files/{file_id}", int(size), **kw)


