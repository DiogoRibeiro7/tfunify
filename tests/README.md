# tfunify Test Suite

This directory contains a comprehensive test suite for the tfunify package, ensuring code quality, correctness, and robustness across all components.

## Test Structure

### Core Test Files

- **`test_core_enhanced.py`** - Comprehensive tests for core mathematical functions
- **`test_european_enhanced.py`** - European TF system tests with edge cases
- **`test_american_enhanced.py`** - American TF system tests including ATR logic
- **`test_tsmom_enhanced.py`** - TSMOM system tests with block structure validation
- **`test_cli_enhanced.py`** - Command-line interface and CSV handling tests
- **`test_integration_enhanced.py`** - Cross-system integration and real-world scenario tests

### Legacy Test Files

- **`test_comprehensive.py`** - Original comprehensive test suite
- **`test_european_basic.py`** - Basic European TF tests
- **`test_smoke.py`** - Quick smoke tests for CI/CD

## Test Categories

### 🧪 Unit Tests

Test individual functions and components in isolation:

- Mathematical correctness of core functions
- Parameter validation and error handling
- Edge cases and boundary conditions
- Input/output shape consistency

### 🔗 Integration Tests

Test interactions between components:

- Cross-system compatibility
- Data pipeline integrity
- Performance across market regimes
- Correlation analysis between systems

### 💻 CLI Tests

Test command-line interface:

- CSV loading and validation
- Command argument parsing
- Error handling and user feedback
- Output file generation

### ⚡ Performance Tests

Test computational efficiency:

- Large dataset handling
- Memory usage optimization
- Execution time benchmarks
- Scalability validation

### 💪 Stress Tests

Test robustness under extreme conditions:

- Market crash scenarios
- High volatility environments
- Numerical edge cases
- Data quality issues

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test category
python run_tests.py unit
python run_tests.py integration
python run_tests.py cli

# Quick smoke test
python run_tests.py --fast
```

### Using pytest directly

```bash
# Run all enhanced tests
pytest tests/test_*_enhanced.py -v

# Run with coverage
pytest --cov=tfunify --cov-report=html

# Run specific test class
pytest tests/test_core_enhanced.py::TestEWMA -v

# Run tests matching pattern
pytest -k "test_volatility" -v
```

### Coverage Analysis

```bash
# Generate coverage report
python run_tests.py coverage

# View HTML coverage report
open htmlcov/index.html
```

## Test Configuration

### pytest.ini

Configuration file with:

- Test discovery settings
- Output formatting options
- Warning filters
- Timeout settings
- Marker definitions

### Markers

Tests are categorized with markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.stress` - Stress tests

## Writing New Tests

### Test Naming Convention

```python
class TestComponentName:
    """Tests for ComponentName."""

    def test_basic_functionality(self):
        """Test basic expected behavior."""
        pass

    def test_edge_case_handling(self):
        """Test edge cases and error conditions."""
        pass

    def test_parameter_validation(self):
        """Test input parameter validation."""
        pass
```

### Test Data Generation

```python
def setup_method(self):
    """Set up test data before each test."""
    np.random.seed(42)  # Reproducible tests
    self.n = 1000
    self.prices = self.generate_test_prices()

def generate_test_prices(self):
    """Generate realistic price data."""
    returns = 0.0002 + 0.015 * np.random.randn(self.n)
    return 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])
```

### Assertion Patterns

```python
# Shape and length checks
assert len(result) == len(input_data)
assert result.shape == expected_shape

# Finite value validation
assert np.isfinite(result[warmup:]).all()

# Approximate equality for floats
np.testing.assert_allclose(result, expected, atol=1e-10)

# Error condition testing
with pytest.raises(ValueError, match="descriptive error message"):
    function_that_should_fail()
```

## Test Data

### Synthetic Data Generation

Tests use controlled synthetic data to ensure:

- Reproducible results (fixed random seeds)
- Known statistical properties
- Specific market scenarios (trending, mean-reverting, volatile)

### Real-World Scenarios

Integration tests simulate:

- Bull and bear markets
- Financial crises
- Low volatility regimes
- Market microstructure effects

## Performance Benchmarks

### Speed Requirements

- Core functions should process 10,000 observations in <1 second
- Full system backtests should complete in reasonable time
- Memory usage should scale linearly with data size

### Accuracy Requirements

- Mathematical functions accurate to machine precision
- Financial calculations match theoretical expectations
- Numerical stability maintained across parameter ranges

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- Pull requests to main/develop branches
- Pushes to protected branches
- Multiple Python versions (3.10, 3.11, 3.12)
- Multiple operating systems

### Coverage Requirements

- Minimum 80% code coverage
- All new features must include tests
- Critical paths require 100% coverage

## Debugging Failed Tests

### Common Issues

1. **Floating Point Precision**

  ```python
  # Use appropriate tolerance
  np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-9)
  ```

2. **Random Seed Inconsistency**

  ```python
  # Always set seed for reproducible tests
  np.random.seed(42)
  ```

3. **Insufficient Warmup Period**

  ```python
  # Allow for system warmup
  warmup = max(config.span_long, config.span_sigma) + 10
  assert np.isfinite(result[warmup:]).all()
  ```

4. **Platform Dependencies**

  ```python
  # Use platform-independent approaches
  assert abs(result - expected) < tolerance
  ```

### Debug Tools

```bash
# Run single test with full output
pytest tests/test_file.py::TestClass::test_method -vvs

# Drop into debugger on failure
pytest --pdb tests/test_file.py

# Show local variables on failure
pytest --tb=long tests/test_file.py
```

## Test Data Files

Some tests may generate temporary files:

- `*_results.npz` - System output files
- `test_data.csv` - Temporary CSV files
- `htmlcov/` - Coverage report directory

These are automatically cleaned up after tests complete.

## Contributing

When adding new tests:

1. **Follow naming conventions** - Use descriptive test names
2. **Include docstrings** - Explain what each test validates
3. **Test edge cases** - Don't just test happy paths
4. **Use appropriate assertions** - Choose the right assertion for the test
5. **Keep tests focused** - One concept per test method
6. **Add markers** - Categorize tests with appropriate markers

### Example Test

```python
def test_ewma_with_extreme_smoothing(self):
    """Test EWMA with very high smoothing parameter."""
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    nu = 0.999  # Very high smoothing

    result = ewma(x, nu)

    # With high smoothing, should stay close to initial value
    assert len(result) == len(x)
    assert result[0] == x[0]  # First value unchanged
    assert np.all(np.abs(result[1:] - x[0]) < 0.1)  # Subsequent values close to first
```

## Performance Optimization

Tests themselves should be efficient:

- Use minimal data sizes that still validate functionality
- Cache expensive setup operations when possible
- Use appropriate random seeds for reproducibility
- Clean up resources after tests

## Conclusion

This comprehensive test suite ensures tfunify maintains high quality and reliability. The tests cover mathematical correctness, edge case handling, performance requirements, and real-world applicability. Regular execution of the full test suite helps maintain confidence in the codebase as it evolves.
