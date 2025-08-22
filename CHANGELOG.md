# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive input validation across all systems
- Enhanced error messages with clear guidance
- Performance benchmarks and regression testing
- Multi-platform CI testing (Ubuntu, Windows, macOS)
- Pre-commit hooks for code quality
- Enhanced CLI with better error handling and user feedback

### Changed

- Improved docstrings with detailed parameter descriptions
- Enhanced README with theory background and Sepp & Lucic (2025) citation
- Updated CI workflows with branch protection and release validation
- Default branch changed to `develop` for better development workflow

### Fixed

- Package structure issues in pyproject.toml
- Import handling for optional dependencies (yfinance)
- Type hints and validation throughout codebase

## [0.1.0] - 2025-01-XX

### Added

- **European TF System**: Variance-preserving EWMA on volatility-normalized returns

  - Single trend filter mode
  - Long-short filter mode with proper variance scaling
  - Volatility targeting for consistent risk exposure
  - Configurable spans for volatility estimation and trend detection

- **American TF System**: Breakout system with ATR buffers and trailing stops

  - Fast/slow moving average crossovers with ATR-based entry thresholds
  - Adaptive position sizing based on ATR and risk multiples
  - Trailing stop mechanism for systematic risk management
  - State machine handling for entry/exit logic

- **TSMOM System**: Time Series Momentum with block-averaged signals

  - Configurable block length (L) and number of blocks (M)
  - Sign-based momentum signals from cumulative returns
  - Volatility normalization and targeting
  - Grid-based signal generation

- **Core Mathematical Functions**:

  - `span_to_nu()`: Convert EWMA spans to smoothing parameters
  - `ewma()`: Exponentially weighted moving average with validation
  - `ewma_variance_preserving()`: Variance-preserving EWMA implementation
  - `long_short_variance_preserving()`: Long-short filter with correct loadings
  - `pct_returns_from_prices()`: Robust percentage return calculation
  - `ewma_volatility_from_returns()`: Volatility estimation from returns
  - `vol_normalised_returns()`: Volatility-normalized return series
  - `volatility_target_weights()`: Position sizing for volatility targeting
  - `volatility_weighted_turnover()`: Turnover calculation

- **Command Line Interface**:

  - `tfu european`: Run European TF system from CSV data
  - `tfu american`: Run American TF system with OHLC data
  - `tfu tsmom`: Run TSMOM system with configurable parameters
  - `tfu download`: Download data from Yahoo Finance (requires extras)
  - Comprehensive parameter configuration for all systems
  - Summary statistics output and detailed results saved to .npz files

- **Data Handling**:

  - CSV loading with automatic high/low fallback to close prices
  - Yahoo Finance integration (optional dependency)
  - Robust error handling for malformed data
  - Input validation with clear error messages

- **Configuration Classes**:

  - `EuropeanTFConfig`: Configuration with validation for European TF
  - `AmericanTFConfig`: Configuration with validation for American TF
  - `TSMOMConfig`: Configuration with validation for TSMOM
  - Post-initialization validation ensuring parameter consistency

- **Development Infrastructure**:

  - Comprehensive test suite with edge case coverage
  - Multi-version Python support (3.10, 3.11, 3.12)
  - Type hints throughout codebase
  - Continuous integration with GitHub Actions
  - Automated release workflow with testing validation
  - Code quality tools (ruff, mypy, pytest)

- **Documentation**:

  - Detailed README with installation and usage examples
  - Theoretical background based on Sepp & Lucic (2025) framework
  - Inline documentation with parameter and return descriptions
  - Examples for all three trend-following systems
  - Development workflow guidelines

### Technical Details

- **Performance**: Pure NumPy implementation optimized for speed
- **Reliability**: Comprehensive input validation and error handling
- **Maintainability**: Type hints, extensive testing, and clear documentation
- **Extensibility**: Modular design allowing easy addition of new systems

### Dependencies

- **Core**: Python >=3.10, NumPy ^2.0.0
- **Optional**: yfinance >=0.2.40 (for data downloading)
- **Development**: pytest, mypy, ruff, pre-commit

--------------------------------------------------------------------------------

## Release Notes Template for Future Versions

When preparing a new release:

1. Move items from `[Unreleased]` to new version section
2. Update version numbers in pyproject.toml and **init**.py
3. Add release date
4. Create git tag: `git tag vX.Y.Z`
5. Push tag to trigger release workflow

### Version Number Guidelines

- **Major (X.0.0)**: Breaking changes, new system architectures
- **Minor (X.Y.0)**: New features, new systems, backwards-compatible
- **Patch (X.Y.Z)**: Bug fixes, documentation, performance improvements
