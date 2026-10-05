# Security policy

## Supported versions

Fixes are made on the latest release.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Write to
dfr@esmad.ipp.pt with a description, the version of tfunify and, if you can,
a way to reproduce it. You will get an answer within a week.

tfunify is a numerical library: it reads the CSV files you give it and, with
the optional `yahoo` extra, downloads prices through `yfinance`. It does not
execute code from its inputs and opens no network connection of its own.
