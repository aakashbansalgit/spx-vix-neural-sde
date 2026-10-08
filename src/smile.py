import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
from bs import callbs, putbs
from config import DEVICE

plt.rcParams["figure.figsize"] = (9, 6)


class Datetime:
    """Thin wrapper around a date string that also stores time-to-maturity in years."""

    def __init__(self, string, T0=None):
        self.string = string
        self.datetime = pd.to_datetime(string)
        self.T0 = T0
        if T0:
            self.t = float((self.datetime - pd.to_datetime(T0)).days / 365)
        else:
            self.t = 0.0

    def __hash__(self):
        return self.string.__hash__()

    def plus_30d(self):
        tau = datetime.timedelta(days=30)
        return self.__class__((self.datetime + tau).strftime("%Y-%m-%d"), T0=self.T0)

    def __str__(self):
        return self.string

    def __repr__(self):
        return self.string.__repr__()

    def __lt__(self, other):
        if isinstance(other, Datetime):
            return self.datetime < other.datetime
        elif isinstance(other, str):
            return self < Datetime(other)
        raise TypeError(f"Cannot compare Datetime with {type(other)}")

    def __le__(self, other):
        if isinstance(other, Datetime):
            return self.datetime <= other.datetime
        elif isinstance(other, str):
            return self <= Datetime(other)
        raise TypeError(f"Cannot compare Datetime with {type(other)}")

    def __gt__(self, other):
        if isinstance(other, Datetime):
            return self.datetime > other.datetime
        elif isinstance(other, str):
            return self > Datetime(other)
        raise TypeError(f"Cannot compare Datetime with {type(other)}")

    def __ge__(self, other):
        if isinstance(other, Datetime):
            return self.datetime >= other.datetime
        elif isinstance(other, str):
            return self >= Datetime(other)
        raise TypeError(f"Cannot compare Datetime with {type(other)}")

    def __eq__(self, other):
        if isinstance(other, Datetime):
            return self.datetime == other.datetime
        elif isinstance(other, str):
            return self.string == other
        raise TypeError(f"Cannot compare Datetime with {type(other)}")


class Smile:
    """Stores a volatility smile — strikes, forward, bid/ask/mid implied vols, and option prices."""

    def __init__(self, instrument, maturity, data):
        self.name = str(instrument)
        self.maturity = maturity
        self.strikes = data["strikes"].clone()
        self.fwd = data["fwd"].clone()
        self.fwd_ask = data["fwd_ask"].clone()
        self.fwd_bid = data["fwd_bid"].clone()
        self.mids = data["mids"].clone()
        self.asks = data["asks"].clone()
        self.bids = data["bids"].clone()
        t = torch.tensor(maturity.t, device=self.fwd.device)
        self.calls = data["calls"] if data["calls"] is not None else callbs(self.fwd, 0.0, t, self.strikes, self.mids)
        self.puts = data["puts"] if data["puts"] is not None else putbs(self.fwd, 0.0, t, self.strikes, self.mids)

    def plot(self, color=None, legend=True):
        with torch.no_grad():
            plt.plot(self.strikes.cpu(), self.mids.cpu(), color=color, label=f"{self.name} mid IV")
            plt.scatter(self.strikes.cpu(), self.mids.cpu(), marker="x", s=20, color=color)
            plt.fill_between(
                self.strikes.cpu(), self.bids.cpu(), self.asks.cpu(),
                color=color, alpha=0.5, label="bid-ask",
            )
            plt.axvline(self.fwd.cpu(), color=color, linestyle="--", label=f"{self.name} fwd")
            plt.xlabel("Strike")
            plt.ylabel("Implied volatility")
            if legend:
                plt.legend()
            plt.title(self.name)

    def to_json(self):
        return {
            "name": self.name,
            "maturity": (self.maturity.T0, self.maturity.string),
            "strikes": self.strikes.detach().cpu().tolist(),
            "fwd": float(self.fwd.detach().cpu()),
            "fwd_bid": float(self.fwd_bid.detach().cpu()),
            "fwd_ask": float(self.fwd_ask.detach().cpu()),
            "mids": self.mids.detach().cpu().tolist(),
            "asks": self.asks.detach().cpu().tolist(),
            "bids": self.bids.detach().cpu().tolist(),
            "calls": self.calls.detach().cpu().tolist(),
            "puts": self.puts.detach().cpu().tolist(),
        }

    @staticmethod
    def from_json(js):
        name = js["name"]
        T0, maturity_str = js["maturity"]
        maturity = Datetime(maturity_str, T0)
        ts = lambda x: torch.tensor(x, device=DEVICE)
        return Smile(
            name, maturity,
            {
                "strikes": ts(js["strikes"]),
                "fwd": ts(js["fwd"]),
                "fwd_bid": ts(js["fwd_bid"]),
                "fwd_ask": ts(js["fwd_ask"]),
                "mids": ts(js["mids"]),
                "asks": ts(js["asks"]),
                "bids": ts(js["bids"]),
                "calls": ts(js["calls"]),
                "puts": ts(js["puts"]),
            },
        )
