try:
    from main import app
    import unittest
    
except Exception as e:
    print("Some Modules are Missing {} ".format(e))
    
class FlaskTest(unittest.TestCase):
    
    
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get("/insertRow")
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)
        
if __name__ == "__main__":
    unittest.main
        