#!/usr/bin/env python3
"""Simple test runner for basic functionality verification."""

import sys
import os
import traceback
from pathlib import Path


def run_basic_tests():
    """Run basic tests without external dependencies."""
    print("🧪 Running basic functionality tests...")
    
    tests_passed = 0
    tests_failed = 0
    
    def test_basic_math():
        """Test basic math operations."""
        assert 1 + 1 == 2
        assert 2 * 3 == 6
        assert 10 / 2 == 5
        return True
    
    def test_python_version():
        """Test Python version."""
        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 9
        return True
    
    def test_environment_setup():
        """Test environment setup."""
        # Set test environment variables
        os.environ['ENVIRONMENT'] = 'testing'
        os.environ['QLOO_API_KEY'] = 'test-key'
        
        assert os.getenv('ENVIRONMENT') == 'testing'
        assert os.getenv('QLOO_API_KEY') == 'test-key'
        return True
    
    def test_directory_operations():
        """Test directory operations."""
        test_dirs = ['data', 'logs', 'reports']
        for directory in test_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            assert Path(directory).exists()
        return True
    
    def test_file_operations():
        """Test file operations."""
        test_file = Path('test_temp.txt')
        test_content = "Hello, World!"
        
        # Write file
        test_file.write_text(test_content)
        
        # Read file
        content = test_file.read_text()
        assert content == test_content
        
        # Clean up
        test_file.unlink()
        assert not test_file.exists()
        return True
    
    def test_json_operations():
        """Test JSON operations."""
        import json
        
        test_data = {"name": "test", "value": 123}
        json_string = json.dumps(test_data)
        parsed_data = json.loads(json_string)
        assert parsed_data == test_data
        return True
    
    def test_basic_imports():
        """Test basic imports."""
        try:
            import json
            import os
            import sys
            import pathlib
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
    
    # List of tests to run
    tests = [
        ("Basic Math", test_basic_math),
        ("Python Version", test_python_version),
        ("Environment Setup", test_environment_setup),
        ("Directory Operations", test_directory_operations),
        ("File Operations", test_file_operations),
        ("JSON Operations", test_json_operations),
        ("Basic Imports", test_basic_imports),
    ]
    
    # Run all tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                tests_passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                tests_failed += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            tests_failed += 1
            traceback.print_exc()
    
    # Summary
    total_tests = tests_passed + tests_failed
    print(f"\n📊 Test Results:")
    print(f"   Total: {total_tests}")
    print(f"   Passed: {tests_passed}")
    print(f"   Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)