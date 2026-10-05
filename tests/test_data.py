"""Reading price files, and the Yahoo download against a stand-in for yfinance."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from tfunify.data import download_csv, load_csv


def write(tmp_path, text, name="prices.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


class TestLoadCsv:
    def test_full_file(self, tmp_path):
        path = write(
            tmp_path,
            "date,open,high,low,close,volume\n"
            "2024-01-02,10,11,9.5,10.5,1000\n"
            "2024-01-03,10.5,12,10,11.5,1500\n",
        )
        data = load_csv(path)
        assert sorted(data) == ["close", "date", "high", "low", "open", "volume"]
        np.testing.assert_array_equal(data["close"], [10.5, 11.5])
        np.testing.assert_array_equal(data["high"], [11.0, 12.0])
        np.testing.assert_array_equal(data["low"], [9.5, 10.0])
        np.testing.assert_array_equal(data["open"], [10.0, 10.5])
        np.testing.assert_array_equal(data["volume"], [1000.0, 1500.0])
        assert data["date"].tolist() == ["2024-01-02", "2024-01-03"]
        assert data["close"].dtype == np.float64

    def test_close_only_file_uses_the_close_for_high_and_low(self, tmp_path):
        data = load_csv(write(tmp_path, "close\n10\n11\n12\n"))
        assert sorted(data) == ["close", "high", "low"]
        np.testing.assert_array_equal(data["high"], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(data["low"], [10.0, 11.0, 12.0])
        data["high"][0] = 99.0  # copies, not views of the close
        assert data["close"][0] == 10.0

    def test_header_is_matched_without_case_and_other_columns_are_ignored(self, tmp_path):
        # the layout of a file saved from Yahoo Finance
        path = write(
            tmp_path,
            "Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2024-01-02,10,11,9,10.5,10.4,100\n"
            "2024-01-03,10.5,12,10,11.5,11.4,200\n",
        )
        data = load_csv(path)
        np.testing.assert_array_equal(data["close"], [10.5, 11.5])
        assert "adj close" not in data

    def test_spaces_blank_lines_and_a_byte_order_mark(self, tmp_path):
        path = write(
            tmp_path, " date , close \n\n2024-01-02, 10 \n,\n2024-01-03,11\n", encoding="utf-8-sig"
        )
        data = load_csv(path)
        np.testing.assert_array_equal(data["close"], [10.0, 11.0])
        assert data["date"].tolist() == ["2024-01-02", "2024-01-03"]

    def test_accepts_a_string_path(self, tmp_path):
        np.testing.assert_array_equal(
            load_csv(str(write(tmp_path, "close\n1\n2\n")))["close"], [1, 2]
        )

    def test_open_and_volume_may_have_gaps(self, tmp_path):
        data = load_csv(write(tmp_path, "open,close,volume\n,10,N/A\n10,11,5\n"))
        assert np.isnan(data["open"][0])
        assert np.isnan(data["volume"][0])
        np.testing.assert_array_equal(data["close"], [10.0, 11.0])

    def test_dates_in_another_format_are_not_checked(self, tmp_path):
        data = load_csv(write(tmp_path, "date,close\n03/01/2024,10\n02/01/2024,11\n"))
        assert data["date"].tolist() == ["03/01/2024", "02/01/2024"]

    def test_timestamps_are_ordered_by_date_and_time(self, tmp_path):
        data = load_csv(
            write(tmp_path, "date,close\n2024-01-02 00:00:00,10\n2024-01-03 00:00:00,11\n")
        )
        assert data["close"].size == 2
        # two prices of the same day are in order if their times are
        text = "date,close\n2024-01-02 09:30,10\n2024-01-02 09:31,11\n2024-01-03T09:30:00,12\n"
        assert load_csv(write(tmp_path, text))["close"].size == 3
        with pytest.raises(ValueError, match=r"row 2 \(2024-01-02 09:30\).*\(2024-01-02 09:31\)"):
            load_csv(write(tmp_path, "date,close\n2024-01-02 09:31,10\n2024-01-02 09:30,11\n"))
        # as Yahoo writes them, with the offset of the exchange
        text = "date,close\n2024-03-08 00:00:00-05:00,10\n2024-03-11 00:00:00-04:00,11\n"
        assert load_csv(write(tmp_path, text))["close"].size == 2
        # stamps with and without an offset cannot be compared: not checked
        text = "date,close\n2024-03-08 00:00:00-05:00,10\n2024-03-07,11\n"
        assert load_csv(write(tmp_path, text))["date"].tolist()[1] == "2024-03-07"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"nowhere\.csv"):
            load_csv(tmp_path / "nowhere.csv")
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path)  # a directory

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "is empty"),
            ("date,open\n2024-01-02,10\n", "no 'close' column; its columns are date, open"),
            ("close\n", "no rows"),
            ("close\n10\nabc\n", r"line 3: column 'close' holds 'abc'"),
            ("close\n10\n\n-5\n", r"line 4: column 'close' holds '-5'"),
            ("close\n10\n0\n", "not a positive number"),
            ("close\n10\nnan\n", "not a positive number"),
            ("close\n10\ninf\n", "not a positive number"),
            ("date,close\n2024-01-02,10\n2024-01-03\n", "line 3: column 'close' holds nothing"),
            ("close,high,low\n10,11,9\n11,,10\n", "line 3: column 'high' holds nothing"),
            ("close,high,low\n10,11,9\n11,10,12\n", "line 3: high 10 is below low 12"),
            ("close,high\n10,11\n", "has a 'high' column but no 'low' column"),
            ("low,close\n9,10\n", "has a 'low' column but no 'high' column"),
            ("close,open,Close\n10,9,10\n", "more than one column named close"),
            ("Date,close,date\n2024-01-02,10,x\n", "more than one column named date"),
            ("date;close\n2024-01-02;10\n", r"its columns are date;close \(the file must be comma"),
            ("date\tclose\n2024-01-02\t10\n", "the file must be comma-separated"),
            (
                "date,close\n2024-01-03,10\n2024-01-02,11\n",
                r"chronological order.*row 2 \(2024-01-02\).*\(2024-01-03\)",
            ),
            ("date,close\n2024-01-02,10\n2024-01-02,11\n", "chronological order"),
        ],
    )
    def test_reports_what_is_wrong_and_where(self, tmp_path, text, message):
        with pytest.raises(ValueError, match=message):
            load_csv(write(tmp_path, text))

    def test_other_columns_may_repeat(self, tmp_path):
        data = load_csv(write(tmp_path, "note,close,note\na,10,b\nc,11,d\n"))
        np.testing.assert_array_equal(data["close"], [10.0, 11.0])

    def test_a_day_without_a_range_is_valid(self, tmp_path):
        data = load_csv(write(tmp_path, "close,high,low\n10,10,10\n11,12,10\n"))
        np.testing.assert_array_equal(data["high"] - data["low"], [0.0, 2.0])

    def test_a_short_row_has_an_empty_date(self, tmp_path):
        # dates that are not all ISO dates are kept as they are, unchecked
        data = load_csv(write(tmp_path, "close,date\n10,2024-01-02\n11\n"))
        assert data["date"].tolist() == ["2024-01-02", ""]

    def test_a_file_that_is_not_text(self, tmp_path):
        path = tmp_path / "prices.csv"
        path.write_bytes("close\n10\n".encode("utf-16"))
        with pytest.raises(ValueError, match="is not a UTF-8 text file"):
            load_csv(path)
        path.write_bytes(b"close\n10\n\xff\xfe11\n")
        with pytest.raises(ValueError, match="is not a UTF-8 text file"):
            load_csv(path)

    def test_a_file_the_csv_reader_rejects(self, tmp_path):
        # errors of the csv module are reported like the others, with the line
        path = tmp_path / "prices.csv"
        path.write_bytes(b"close\n10\n" + b"1" * 200_000 + b"\n")
        with pytest.raises(ValueError, match="line 3: field larger than field limit"):
            load_csv(path)
        path.write_bytes(b"close\n10\n1\x001\n")  # csv.Error before Python 3.11
        with pytest.raises(ValueError, match="line 3"):
            load_csv(path)


@pytest.fixture
def fake_yfinance(monkeypatch):
    """Install a stand-in ``yfinance`` whose ``download`` returns a prepared frame."""
    pd = pytest.importorskip("pandas")
    calls = []
    module = types.ModuleType("yfinance")

    def download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return module.frame

    module.download = download
    module.frame = None
    module.calls = calls
    module.pd = pd
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return module


def frame(pd, columns, rows, dates=("2024-01-02", "2024-01-03", "2024-01-04")):
    return pd.DataFrame(rows, columns=columns, index=pd.to_datetime(list(dates[: len(rows)])))


ROWS = [[10.0, 11.0, 9.0, 10.5, 100], [10.5, 12.0, 10.0, 11.5, 200], [11.5, 12.5, 11.0, 12.0, 300]]
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class TestDownloadCsv:
    def test_writes_a_file_that_load_csv_reads(self, tmp_path, fake_yfinance):
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, ROWS)
        out = download_csv("SPY", tmp_path / "spy.csv", period="1y")
        assert out == tmp_path / "spy.csv"
        assert out.read_text().splitlines()[:2] == [
            "date,open,high,low,close,volume",
            "2024-01-02,10.0,11.0,9.0,10.5,100",
        ]
        data = load_csv(out)
        np.testing.assert_array_equal(data["close"], [10.5, 11.5, 12.0])
        np.testing.assert_array_equal(data["high"], [11.0, 12.0, 12.5])
        assert fake_yfinance.calls == [
            ("SPY", {"period": "1y", "interval": "1d", "auto_adjust": False, "progress": False})
        ]

    def test_two_level_columns_of_recent_yfinance(self, tmp_path, fake_yfinance):
        pd = fake_yfinance.pd
        columns = pd.MultiIndex.from_product([COLUMNS, ["SPY"]], names=["Price", "Ticker"])
        fake_yfinance.frame = frame(pd, columns, ROWS)
        data = load_csv(download_csv("SPY", tmp_path / "spy.csv"))
        np.testing.assert_array_equal(data["close"], [10.5, 11.5, 12.0])
        np.testing.assert_array_equal(data["volume"], [100.0, 200.0, 300.0])

    def test_two_level_columns_with_the_ticker_first(self, tmp_path, fake_yfinance):
        pd = fake_yfinance.pd
        columns = pd.MultiIndex.from_product([["SPY"], COLUMNS])
        fake_yfinance.frame = frame(pd, columns, ROWS)
        data = load_csv(download_csv("SPY", tmp_path / "spy.csv"))
        np.testing.assert_array_equal(data["low"], [9.0, 10.0, 11.0])

    def test_two_level_columns_without_prices(self, tmp_path, fake_yfinance):
        pd = fake_yfinance.pd
        columns = pd.MultiIndex.from_product([["Bid", "Ask"], ["SPY"]])
        fake_yfinance.frame = frame(pd, columns, [[1.0, 2.0]])
        with pytest.raises(ValueError, match=r"lacks the column\(s\) Open, High, Low, Close"):
            download_csv("SPY", tmp_path / "x.csv")

    def test_incomplete_rows_are_skipped_and_missing_volume_is_zero(self, tmp_path, fake_yfinance):
        rows = [
            ROWS[0],
            [10.5, 12.0, float("nan"), 11.5, 200],
            [11.5, 12.5, 11.0, 12.0, float("nan")],
        ]
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, rows)
        data = load_csv(download_csv("SPY", tmp_path / "spy.csv"))
        assert data["date"].tolist() == ["2024-01-02", "2024-01-04"]
        np.testing.assert_array_equal(data["volume"], [100.0, 0.0])

    def test_rows_with_a_price_that_is_not_positive_are_skipped(self, tmp_path, fake_yfinance):
        # what load_csv would refuse must not be written
        rows = [ROWS[0], [10.5, 12.0, 0.0, 11.5, 200], [11.5, 12.5, 11.0, -12.0, 300], ROWS[2]]
        dates = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, rows, dates)
        data = load_csv(download_csv("SPY", tmp_path / "spy.csv"))
        assert data["date"].tolist() == ["2024-01-02", "2024-01-05"]

    def test_no_volume_column(self, tmp_path, fake_yfinance):
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS[:4], [row[:4] for row in ROWS])
        data = load_csv(download_csv("EURUSD=X", tmp_path / "fx.csv"))
        np.testing.assert_array_equal(data["volume"], [0.0, 0.0, 0.0])

    def test_adjusted_prices_are_requested_on_demand(self, tmp_path, fake_yfinance):
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, ROWS)
        download_csv("SPY", tmp_path / "spy.csv", interval="1wk", auto_adjust=True)
        assert fake_yfinance.calls[0][1]["auto_adjust"] is True
        assert fake_yfinance.calls[0][1]["interval"] == "1wk"

    def test_no_data(self, tmp_path, fake_yfinance):
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, [])
        with pytest.raises(ValueError, match="no data for 'NOPE'"):
            download_csv("NOPE", tmp_path / "x.csv")
        fake_yfinance.frame = None
        with pytest.raises(ValueError, match="no data for 'NOPE'"):
            download_csv("NOPE", tmp_path / "x.csv")
        assert not (tmp_path / "x.csv").exists()

    def test_no_complete_rows(self, tmp_path, fake_yfinance):
        rows = [[float("nan")] * 5, [float("nan")] * 5]
        fake_yfinance.frame = frame(fake_yfinance.pd, COLUMNS, rows)
        with pytest.raises(ValueError, match="no complete rows"):
            download_csv("SPY", tmp_path / "x.csv")

    def test_missing_price_columns(self, tmp_path, fake_yfinance):
        fake_yfinance.frame = frame(fake_yfinance.pd, ["Open", "Close"], [[1.0, 2.0]])
        with pytest.raises(ValueError, match=r"lacks the column\(s\) High, Low"):
            download_csv("SPY", tmp_path / "x.csv")

    def test_invalid_interval(self, tmp_path):
        with pytest.raises(ValueError, match="interval"):
            download_csv("SPY", tmp_path / "x.csv", interval="5m")

    def test_without_yfinance_the_error_says_how_to_install_it(self, tmp_path, monkeypatch):
        monkeypatch.delitem(sys.modules, "yfinance", raising=False)
        real_import = builtins.__import__

        def no_yfinance(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("No module named 'yfinance'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_yfinance)
        with pytest.raises(ImportError, match=r'pip install "tfunify\[yahoo\]"'):
            download_csv("SPY", tmp_path / "x.csv")
