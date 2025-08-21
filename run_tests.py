#!/usr/bin/env python3
"""
Comprehensive test runner for tfunify package.

This script runs the full test suite with different configurations
and provides detailed reporting on test coverage and performance.
"""

import argparse
import subprocess
import sys
import time


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'=' * 60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    print(f"Exit code: {result.returncode}")
    print(f"Duration: {duration:.2f}s")

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_core_enhanced.py",
        "tests/test_european_enhanced.py",
        "tests/test_american_enhanced.py",
        "tests/test_tsmom_enhanced.py",
        "-v",
        "--tb=short",
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_integration_enhanced.py",
        "tests/test_comprehensive.py",
        "-v",
        "--tb=short",
    ]
    return run_command(cmd, "Integration Tests")


def run_cli_tests():
    """Run CLI tests."""
    cmd = ["python", "-m", "pytest", "tests/test_cli_enhanced.py", "-v", "--tb=short"]
    return run_command(cmd, "CLI Tests")


def run_coverage_tests():
    """Run tests with coverage reporting."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "--cov=tfunify",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=80",
        "-v",
    ]
    return run_command(cmd, "Coverage Tests")


def run_performance_tests():
    """Run performance-focused tests."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_integration_enhanced.py::TestCrossSystemIntegration::test_all_systems_run_successfully",
        "tests/test_integration_enhanced.py::TestRealWorldScenarios",
        "-v",
        "--durations=0",
    ]
    return run_command(cmd, "Performance Tests")


def run_stress_tests():
    """Run stress tests."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_integration_enhanced.py::TestCrossSystemIntegration::test_stress_testing",
        "tests/test_core_enhanced.py::TestEWMA::test_extreme_nu_values",
        "tests/test_european_enhanced.py::TestEuropeanTF::test_extreme_parameters",
        "tests/test_american_enhanced.py::TestAmericanTF::test_extreme_market_conditions",
        "tests/test_tsmom_enhanced.py::TestTSMOM::test_extreme_parameter_combinations",
        "-v",
        "--tb=short",
    ]
    return run_command(cmd, "Stress Tests")


def run_smoke_tests():
    """Run quick smoke tests."""
    cmd = ["python", "-m", "pytest", "tests/test_smoke.py", "tests/test_european_basic.py", "-v"]
    return run_command(cmd, "Smoke Tests")


def run_type_checking():
    """Run mypy type checking."""
    cmd = ["python", "-m", "mypy", "tfunify", "--strict"]
    return run_command(cmd, "Type Checking with MyPy")


def run_linting():
    """Run code linting."""
    results = []

    # Ruff check
    cmd = ["python", "-m", "ruff", "check", "tfunify", "tests"]
    results.append(run_command(cmd, "Ruff Linting"))

    # Ruff format check
    cmd = ["python", "-m", "ruff", "format", "--check", "tfunify", "tests"]
    results.append(run_command(cmd, "Ruff Format Check"))

    return results


def run_all_tests():
    """Run the complete test suite."""
    print("=" * 80)
    print("TFUNIFY COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    results = {}
    total_start = time.time()

    # 1. Smoke tests first
    print("\n🔥 SMOKE TESTS")
    results["smoke"] = run_smoke_tests()

    # 2. Unit tests
    print("\n🧪 UNIT TESTS")
    results["unit"] = run_unit_tests()

    # 3. Integration tests
    print("\n🔗 INTEGRATION TESTS")
    results["integration"] = run_integration_tests()

    # 4. CLI tests
    print("\n💻 CLI TESTS")
    results["cli"] = run_cli_tests()

    # 5. Performance tests
    print("\n⚡ PERFORMANCE TESTS")
    results["performance"] = run_performance_tests()

    # 6. Stress tests
    print("\n💪 STRESS TESTS")
    results["stress"] = run_stress_tests()

    # 7. Code quality checks
    print("\n🎯 CODE QUALITY CHECKS")
    results["linting"] = run_linting()
    results["typing"] = run_type_checking()

    # 8. Coverage tests (last, as they're comprehensive)
    print("\n📊 COVERAGE TESTS")
    results["coverage"] = run_coverage_tests()

    total_duration = time.time() - total_start

    # Summary report
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {total_duration:.2f}s")
    print()

    success_count = 0
    total_count = 0

    for test_type, result in results.items():
        if isinstance(result, list):  # Multiple results (like linting)
            for i, res in enumerate(result):
                status = "✅ PASS" if res.returncode == 0 else "❌ FAIL"
                print(f"{test_type}[{i}]: {status}")
                total_count += 1
                if res.returncode == 0:
                    success_count += 1
        else:
            status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
            print(f"{test_type}: {status}")
            total_count += 1
            if result.returncode == 0:
                success_count += 1

    print(f"\nOverall: {success_count}/{total_count} test categories passed")

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tfunify test suite")
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=[
            "all",
            "unit",
            "integration",
            "cli",
            "coverage",
            "performance",
            "stress",
            "smoke",
            "lint",
            "type",
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run only smoke tests for quick feedback"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.fast:
        print("Running fast smoke tests...")
        result = run_smoke_tests()
        return result.returncode

    # Run specific test type
    test_functions = {
        "all": run_all_tests,
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "cli": run_cli_tests,
        "coverage": run_coverage_tests,
        "performance": run_performance_tests,
        "stress": run_stress_tests,
        "smoke": run_smoke_tests,
        "lint": lambda: run_linting()[0],  # Return first linting result
        "type": run_type_checking,
    }

    test_func = test_functions[args.test_type]

    if args.test_type == "all":
        return test_func()
    else:
        result = test_func()
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
