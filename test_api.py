# test_api.py
import httpx
import json
import sys

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "="*50)
    print("TEST: Health Check")
    print("="*50)
    
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    return True


def test_single_prediction():
    """Test single complaint prediction."""
    print("\n" + "="*50)
    print("TEST: Single Prediction")
    print("="*50)
    
    payload = {
        "complaint_text": "Loud music party next door every night, unbearable noise",
        "borough": "BROOKLYN",
        "zip_code": "11201"
    }
    
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result["status"] == "success"
    assert "department" in result
    assert "sentiment" in result
    assert "priority" in result
    
    return True


def test_critical_complaint():
    """Test critical/urgent complaint."""
    print("\n" + "="*50)
    print("TEST: Critical Complaint")
    print("="*50)
    
    payload = {
        "complaint_text": "Dangerous gas leak on main road, people evacuating, emergency situation",
        "borough": "MANHATTAN"
    }
    
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    # Critical complaints should have high priority
    assert result["priority"]["priority_score"] >= 3.0
    
    return True


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*50)
    print("TEST: Batch Prediction")
    print("="*50)
    
    complaints = [
        {"complaint_text": "Blocked driveway, no access to garage"},
        {"complaint_text": "Thank you, the issue has been resolved quickly"},
        {"complaint_text": "Illegal parking blocking fire hydrant"},
        {"complaint_text": "Graffiti on public building wall"},
    ]
    
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json=complaints
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total: {result['total']}")
    print(f"Successful: {result['successful']}")
    print(f"Results: {json.dumps(result['results'], indent=2)}")
    
    assert response.status_code == 200
    assert result["total"] == 4
    assert result["successful"] == 4
    
    return True


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*50)
    print("TEST: Model Info")
    print("="*50)
    
    response = httpx.get(f"{BASE_URL}/models/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    
    return True


def test_validation_error():
    """Test input validation."""
    print("\n" + "="*50)
    print("TEST: Validation Error")
    print("="*50)
    
    payload = {
        "complaint_text": "ab"  # Too short
    }
    
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*50)
    print("# RUNNING API TESTS")
    print("#"*50)
    
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Critical Complaint", test_critical_complaint),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Validation Error", test_validation_error),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            results.append((name, f"FAIL: {str(e)}"))
            print(f"Error: {e}")
    
    # Summary
    print("\n" + "#"*50)
    print("# TEST SUMMARY")
    print("#"*50)
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {name}: {result}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)