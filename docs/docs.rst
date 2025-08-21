tfunify: Unified Trend-Following Systems
=====================================

.. image:: https://img.shields.io/pypi/v/tfunify.svg
   :target: https://pypi.org/project/tfunify/
   :alt: PyPI version

.. image:: https://img.shields.io/codecov/c/github/DiogoRibeiro7/tfunify
   :target: https://codecov.io/gh/DiogoRibeiro7/tfunify
   :alt: Coverage

.. image:: https://readthedocs.org/projects/tfunify/badge/?version=latest
   :target: https://tfunify.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**tfunify** is a comprehensive Python library for trend-following systems based on the unified framework presented in Sepp & Lucic (2025). It implements three distinct trend-following approaches with robust validation and professional-grade analysis tools.

Features
--------

* **European TF**: Variance-preserving EWMA on volatility-normalized returns
* **American TF**: Breakout systems with ATR buffers and trailing stops  
* **TSMOM**: Time Series Momentum with block-averaged signals
* **Comprehensive validation**: Input validation with clear error messages
* **Performance analysis**: Risk-adjusted metrics and visualization
* **Portfolio integration**: Tools for portfolio construction and overlay strategies

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from tfunify import EuropeanTF, EuropeanTFConfig

   # Generate sample data
   np.random.seed(0)
   prices = 100 * np.cumprod(1 + 0.01 * np.random.randn(1000))

   # Configure and run system
   config = EuropeanTFConfig(sigma_target_annual=0.15)
   system = EuropeanTF(config)
   pnl, weights, signal, volatility = system.run_from_prices(prices)

Installation
------------

.. code-block:: bash

   pip install tfunify

   # With optional dependencies
   pip install tfunify[yahoo]

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   theory
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/european
   api/american
   api/tsmom
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
