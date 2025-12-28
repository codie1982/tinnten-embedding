from app import app
import unittest
import json

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_routes_exist(self):
        # We just want to check that we don't get a 404 for these paths.
        # Since we might not have auth/db setup, a 401 or 500 or 400 is fine, as long as it's not 404.
        
        paths = [
            "/categories",
            "/embedding/categories",
            "/api/v10/categories"
        ]
        
        for p in paths:
            # Test POST (create)
            print(f"Testing POST {p}")
            resp = self.app.post(p, json={})
            # If route didn't exist, we'd get 404.
            # If it exists but we lack auth/params, we get 401/400/500/etc.
            self.assertNotEqual(resp.status_code, 404, f"Path {p} returned 404")

if __name__ == '__main__':
    unittest.main()
