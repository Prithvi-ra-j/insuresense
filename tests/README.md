# 🧪 InsureSense 360 Testing Suite

This folder contains all the testing files for InsureSense 360. **Run these tests in order before deploying locally** to ensure everything is working correctly.

## 📋 Prerequisites

Before running tests, ensure you have:

1. **Python 3.8+** installed
2. **All dependencies** installed: `pip install -r requirements.txt`
3. **Environment variables** configured (see `env_template.txt`)
4. **Together AI API key** set in your `.env` file

## 🚀 Testing Sequence

**IMPORTANT: Run these tests in the exact order shown below for best results.**

### **Option 1: Run All Tests at Once (Recommended)**
```bash
# From project root directory
python run_tests.py
```

### **Option 2: Run Tests Individually**

### 1. **Basic System Test**
```bash
python tests/test_system.py
```
**Purpose**: Verifies basic system setup and dependencies
**Expected**: All system components should initialize successfully
**Status**: ✅ Should pass

### 2. **LLM Configuration Test**
```bash
python tests/test_llm.py
```
**Purpose**: Tests basic LLM configuration and initialization
**Expected**: LLM should initialize without errors
**Status**: ✅ Should pass

### 3. **Direct LLM Test**
```bash
python tests/test_direct_llm.py
```
**Purpose**: Tests direct LLM response generation
**Expected**: LLM should generate responses to simple queries
**Status**: ✅ Should pass

### 4. **Together AI Integration Test**
```bash
python tests/test_together_llm.py
```
**Purpose**: Comprehensive test of Together AI + Llama integration
**Expected**: All tests should pass (configuration, initialization, response, RAG)
**Status**: ✅ Should pass (may have rate limit warnings)

### 5. **API Endpoints Test**
```bash
python tests/test_api.py
```
**Purpose**: Tests individual API endpoints
**Expected**: All endpoints should respond correctly
**Status**: ✅ Should pass

### 6. **API Integration Test**
```bash
python tests/test_api_integration.py
```
**Purpose**: End-to-end API integration testing
**Expected**: Complete API workflow should work
**Status**: ✅ Should pass

## 📊 Test Results Interpretation

### ✅ **All Tests Pass**
- Your system is ready for deployment
- All components are working correctly
- You can proceed with local deployment

### ⚠️ **Some Tests Fail**
- Check error messages for specific issues
- Verify environment variables are set correctly
- Ensure all dependencies are installed
- Check API key configuration

### ❌ **Multiple Tests Fail**
- Review the setup process
- Check Python version compatibility
- Verify file permissions
- Contact support if issues persist

## 🔧 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**
   - Check your `.env` file
   - Verify Together AI API key is valid
   - Ensure no extra spaces in the key

3. **Rate Limit Warnings**
   - These are normal for free tier
   - System will use fallback responses
   - Not a critical issue

4. **Import Errors**
   - Ensure you're in the project root directory
   - Check Python path configuration
   - Verify all files are in correct locations

## 📁 Test Files Overview

| File | Purpose | Dependencies |
|------|---------|--------------|
| `test_system.py` | Basic system verification | None |
| `test_llm.py` | LLM configuration test | Together AI |
| `test_direct_llm.py` | Direct LLM testing | Together AI |
| `test_together_llm.py` | Comprehensive LLM test | Together AI, Vector Store |
| `test_api.py` | API endpoint testing | FastAPI, Server |
| `test_api_integration.py` | End-to-end API test | All components |

## 🎯 Success Criteria

After running all tests, you should see:

- ✅ **System initialization** successful
- ✅ **LLM configuration** working
- ✅ **Vector store** accessible
- ✅ **API endpoints** responding
- ✅ **RAG chatbot** functional
- ✅ **Document retrieval** working

## 🚀 Next Steps

Once all tests pass:

1. **Start the server**: `python start_server.py`
2. **Test the frontend**: Navigate to the web interface
3. **Upload policies**: Test policy upload functionality
4. **Chat with the bot**: Test RAG chatbot responses

## 🏃‍♂️ Quick Test Runner

For convenience, you can run all tests at once using the test runner:

```bash
# From project root directory
python run_tests.py
```

This will:
- Run all tests in the correct order
- Show detailed results for each test
- Provide a summary of all test results
- Exit with appropriate status code

## 📞 Support

If you encounter issues:

1. Check the error messages carefully
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Check the main project README
5. Open an issue with detailed error information

---

**Remember: Always run these tests before deploying to ensure a smooth experience!** 🚀
