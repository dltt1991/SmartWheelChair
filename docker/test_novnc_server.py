#!/usr/bin/env python3
import unittest

from novnc_server import NoCacheProxyRequestHandler


class NoVncCacheHeadersTest(unittest.TestCase):
    def test_static_responses_disable_browser_cache(self):
        handler = NoCacheProxyRequestHandler.__new__(NoCacheProxyRequestHandler)
        handler.request_version = "HTTP/1.1"
        handler._headers_buffer = []
        handler.flush_headers = lambda: None

        handler.end_headers()

        headers = b"".join(handler._headers_buffer)
        self.assertIn(b"Cache-Control: no-store", headers)


if __name__ == "__main__":
    unittest.main()
