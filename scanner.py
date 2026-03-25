"""
Kalshi Weather Edge Scanner
============================
Finds mispriced weather markets on Kalshi by comparing crowd odds
against NOAA/NWS forecast model probabilities.

Setup:
    pip install requests python-dotenv colorama tabulate

Usage:
    python scanner.py                    # Scan all weather markets
    python scanner.py --city "Las Vegas" # Filter by city
    python scanner.py --min-edge 0.08   # Only show 8%+ edge
    python scanner.py --top 9           # Show top 9 bets for sequential queue
    python scanner.py --auto-queue      # Auto-build optimal 9-bet queue
"""

import os
import sys
import json
import time
import math
import argparse
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
NOAA_API_BASE   = "https://api.weather.gov"
NOAA_USER_AGENT = "KalshiWeatherBot/1.0 (your@email.com)"  # NOAA requires a UA string

# Kalshi credentials — set in .env or environment
KALSHI_EMAIL    = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.getenv("KALSHI_PASSWORD", "")
KALSHI_API_KEY  = os.getenv("KALSHI_API_KEY", "")  # Alternative: API key auth

# Edge thresholds
STRONG_EDGE  = 0.12   # 12%+ = strong signal
MEDIUM_EDGE  = 0.07   # 7%+  = medium signal
MIN_EDGE     = 0.05   # 5%+  = minimum worth noting

# Known weather station coordinates for major US cities
# Format: city -> (lat, lon, NOAA grid office, grid X, grid Y)
CITY_GRID_MAP = {
    "Las Vegas":    (36.1699, -115.1398, "VEF", 35, 35),
    "New York":     (40.7128,  -74.0060, "OKX", 33, 37),
    "Los Angeles":  (34.0522, -118.2437, "LOX", 74, 55),
    "Chicago":      (41.8781,  -87.6298, "LOT", 74, 70),
    "Phoenix":      (33.4484, -112.0740, "PSR", 53, 31),
    "Houston":      (29.7604,  -95.3698, "HGX", 67, 60),
    "Miami":        (25.7617,  -80.1918, "MFL", 110, 39),
    "Denver":       (39.7392, -104.9903, "BOU", 57, 60),
    "Seattle":      (47.6062, -122.3321, "SEW", 124, 67),
    "Boston":       (42.3601,  -71.0589, "BOX", 64, 54),
    "Atlanta":      (33.7490,  -84.3880, "FFC", 53, 56),
    "Dallas":       (32.7767,  -96.7970, "FWD", 79, 82),
    "Minneapolis":  (44.9778,  -93.2650, "MPX", 106, 70),
    "Kansas City":  (39.0997,  -94.5786, "EAX", 48, 51),
    "New Orleans":  (29.9511,  -90.0715, "LIX", 66, 67),
    "Portland":     (45.5051, -122.6750, "PQR", 110, 103),
    "Detroit":      (42.3314,  -83.0458, "DTX", 65, 64),
    "Nashville":    (36.1627,  -86.7816, "OHX", 27, 67),
    "San Francisco":(37.7749, -122.4194, "MTR", 85, 105),
    "Orlando":      (28.5383,  -81.3792, "MLB", 56, 103),
}


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class WeatherMarket:
    ticker:        str
    title:         str
    category:      str           # temp / rain / snow / wind / storm
    city:          str
    kalshi_yes_prob: float       # Kalshi crowd implied probability for YES
    noaa_prob:     float         # NOAA model probability for YES
    edge:          float         # noaa_prob - kalshi_yes_prob
    direction:     str           # "YES" or "NO" (which side has edge)
    net_edge:      float         # absolute edge magnitude
    resolution_dt: Optional[str]
    volume:        int           # dollar volume
    open_interest: int
    noaa_source:   str           # which NOAA product was used
    reasoning:     str           # human-readable explanation


@dataclass
class BetQueue:
    bets:          list = field(default_factory=list)
    start_capital: float = 200.0

    def add(self, market: WeatherMarket):
        if len(self.bets) < 9:
            self.bets.append(market)

    def projected_payout(self):
        """Payout assuming 2:1 odds and all bets win (theoretical max)."""
        amount = self.start_capital
        for _ in self.bets:
            amount *= 2
        return amount

    def win_probability(self):
        """Compound probability of winning all queued bets."""
        prob = 1.0
        for bet in self.bets:
            # Use NOAA probability as our best estimate of true win prob
            win_prob = bet.noaa_prob if bet.direction == "YES" else (1 - bet.noaa_prob)
            prob *= win_prob
        return prob


# ─────────────────────────────────────────────
# Kalshi API Client
# ─────────────────────────────────────────────

class KalshiClient:
    def __init__(self):
        self.session = requests.Session()
        self.token   = None
        self.base    = KALSHI_API_BASE

    def login(self) -> bool:
        """Authenticate with email/password and store JWT token."""
        if not KALSHI_EMAIL or not KALSHI_PASSWORD:
            cprint("⚠  No Kalshi credentials found. Set KALSHI_EMAIL and KALSHI_PASSWORD in .env", "yellow")
            cprint("   Running in demo mode with simulated market data.", "yellow")
            return False
        try:
            resp = self.session.post(
                f"{self.base}/login",
                json={"email": KALSHI_EMAIL, "password": KALSHI_PASSWORD},
                timeout=10
            )
            resp.raise_for_status()
            self.token = resp.json().get("token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            cprint("✓  Authenticated with Kalshi", "green")
            return True
        except Exception as e:
            cprint(f"✗  Kalshi login failed: {e}", "red")
            return False

    def get_weather_markets(self, limit: int = 200) -> list[dict]:
        """
        Fetch active weather-related markets from Kalshi.
        Uses the /markets endpoint filtered by weather-related series.
        """
        markets = []
        weather_keywords = [
            "temperature", "rain", "rainfall", "snow", "snowfall",
            "wind", "storm", "hurricane", "tornado", "precipitation",
            "high temp", "low temp", "degrees", "inches"
        ]
        try:
            params = {
                "limit":  limit,
                "status": "open",
            }
            resp = self.session.get(f"{self.base}/markets", params=params, timeout=15)
            resp.raise_for_status()
            all_markets = resp.json().get("markets", [])

            for m in all_markets:
                title = (m.get("title") or "").lower()
                subtitle = (m.get("subtitle") or "").lower()
                text = title + " " + subtitle
                if any(kw in text for kw in weather_keywords):
                    markets.append(m)

            cprint(f"✓  Found {len(markets)} weather markets on Kalshi", "green")
        except Exception as e:
            cprint(f"✗  Failed to fetch Kalshi markets: {e}", "red")
            cprint("   Falling back to demo data...", "yellow")
        return markets

    def get_orderbook(self, ticker: str) -> Optional[dict]:
        """Get live orderbook for a market to compute mid-market probability."""
        try:
            resp = self.session.get(
                f"{self.base}/markets/{ticker}/orderbook",
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def place_order(self, ticker: str, side: str, count: int, price: int) -> Optional[dict]:
        """
        Place a limit order on Kalshi.
        side  : "yes" or "no"
        count : number of contracts
        price : cents (1-99)
        """
        if not self.token:
            cprint("✗  Not authenticated — cannot place order", "red")
            return None
        try:
            payload = {
                "ticker":       ticker,
                "client_order_id": f"wbot_{int(time.time())}",
                "type":         "limit",
                "action":       "buy",
                "side":         side,
                "count":        count,
                "yes_price":    price if side == "yes" else (100 - price),
                "no_price":     price if side == "no"  else (100 - price),
            }
            resp = self.session.post(
                f"{self.base}/portfolio/orders",
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            cprint(f"✗  Order failed: {e}", "red")
            return None


# ─────────────────────────────────────────────
# NOAA API Client
# ─────────────────────────────────────────────

class NOAAClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": NOAA_USER_AGENT})
        self.base    = NOAA_API_BASE
        self._cache  = {}

    def get_forecast(self, city: str) -> Optional[dict]:
        """
        Fetch the full NWS forecast for a city.
        Returns the raw forecast JSON or None.
        """
        if city in self._cache:
            return self._cache[city]

        city_data = CITY_GRID_MAP.get(city)
        if not city_data:
            # Try to look up the grid point dynamically
            return self._lookup_and_fetch(city)

        lat, lon, office, grid_x, grid_y = city_data
        url = f"{self.base}/gridpoints/{office}/{grid_x},{grid_y}/forecast"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._cache[city] = data
            return data
        except Exception as e:
            cprint(f"  ✗ NOAA forecast failed for {city}: {e}", "red")
            return None

    def get_hourly_forecast(self, city: str) -> Optional[dict]:
        """Fetch hourly forecast grid data (more granular)."""
        city_data = CITY_GRID_MAP.get(city)
        if not city_data:
            return None
        lat, lon, office, grid_x, grid_y = city_data
        url = f"{self.base}/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def get_gridpoint_data(self, city: str) -> Optional[dict]:
        """
        Fetch raw gridpoint data — includes quantitative forecast values
        like temperature, wind speed, precipitation probability as arrays.
        This is the most useful endpoint for probability calculations.
        """
        city_data = CITY_GRID_MAP.get(city)
        if not city_data:
            return None
        lat, lon, office, grid_x, grid_y = city_data
        url = f"{self.base}/gridpoints/{office}/{grid_x},{grid_y}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            cprint(f"  ✗ NOAA gridpoint data failed for {city}: {e}", "red")
            return None

    def _lookup_and_fetch(self, city_name: str) -> Optional[dict]:
        """
        Dynamically look up a city's grid point using the /points endpoint.
        Requires a rough lat/lon — this is a fallback.
        """
        # We don't have lat/lon for unknown cities, so we skip gracefully
        cprint(f"  ℹ  City '{city_name}' not in local map — add to CITY_GRID_MAP", "yellow")
        return None

    def extract_precip_probability(self, forecast_data: dict, hours_ahead: int = 24) -> Optional[float]:
        """
        Parse precipitation probability from NWS forecast periods.
        Returns probability as 0.0-1.0 for the next N hours.
        """
        if not forecast_data:
            return None
        try:
            periods = forecast_data["properties"]["periods"]
            probs = []
            for period in periods[:max(2, hours_ahead // 12)]:
                pchance = period.get("probabilityOfPrecipitation", {})
                val = pchance.get("value")
                if val is not None:
                    probs.append(val / 100.0)
            return max(probs) if probs else None
        except (KeyError, TypeError):
            return None

    def extract_max_temp(self, forecast_data: dict) -> Optional[float]:
        """Extract the forecasted maximum temperature (°F) for today/tomorrow."""
        if not forecast_data:
            return None
        try:
            periods = forecast_data["properties"]["periods"]
            for period in periods[:4]:
                if period.get("isDaytime", True):
                    temp = period.get("temperature")
                    unit = period.get("temperatureUnit", "F")
                    if temp is not None:
                        if unit == "C":
                            temp = temp * 9/5 + 32
                        return float(temp)
            return None
        except (KeyError, TypeError):
            return None

    def extract_min_temp(self, forecast_data: dict) -> Optional[float]:
        """Extract the forecasted minimum temperature (°F)."""
        if not forecast_data:
            return None
        try:
            periods = forecast_data["properties"]["periods"]
            for period in periods[:4]:
                if not period.get("isDaytime", True):
                    temp = period.get("temperature")
                    unit = period.get("temperatureUnit", "F")
                    if temp is not None:
                        if unit == "C":
                            temp = temp * 9/5 + 32
                        return float(temp)
            return None
        except (KeyError, TypeError):
            return None

    def extract_max_wind(self, forecast_data: dict) -> Optional[float]:
        """Extract max wind speed (mph) from forecast."""
        if not forecast_data:
            return None
        try:
            periods = forecast_data["properties"]["periods"]
            winds = []
            for period in periods[:4]:
                wind_str = period.get("windSpeed", "")
                if "mph" in wind_str:
                    parts = wind_str.replace(" mph", "").split(" to ")
                    for p in parts:
                        try:
                            winds.append(float(p.strip()))
                        except ValueError:
                            pass
            return max(winds) if winds else None
        except (KeyError, TypeError):
            return None

    def temp_threshold_probability(self, forecast_temp: float, threshold: float,
                                    direction: str = "above", uncertainty_std: float = 4.0) -> float:
        """
        Convert a deterministic forecast temperature into a probability.
        Uses a normal distribution centered on the forecast with given uncertainty.

        direction: "above" (P(temp > threshold)) or "below" (P(temp < threshold))
        uncertainty_std: NWS forecast error std dev in °F (typically 3-5°F at 24hr)
        """
        z = (forecast_temp - threshold) / uncertainty_std
        # CDF of standard normal
        prob_above = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return prob_above if direction == "above" else (1 - prob_above)

    def precip_amount_probability(self, forecast_precip_in: float, threshold: float,
                                   uncertainty_std: float = 0.3) -> float:
        """
        Convert a deterministic precip forecast (inches) into P(precip > threshold).
        """
        if forecast_precip_in is None:
            return 0.5  # Unknown — return neutral
        z = (forecast_precip_in - threshold) / max(uncertainty_std, 0.01)
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def wind_threshold_probability(self, forecast_wind: float, threshold: float,
                                    uncertainty_std: float = 5.0) -> float:
        """P(wind > threshold mph)"""
        if forecast_wind is None:
            return 0.5
        z = (forecast_wind - threshold) / uncertainty_std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ─────────────────────────────────────────────
# Market Parser — Extracts Structured Info
# ─────────────────────────────────────────────

class MarketParser:
    """Parses Kalshi market titles to extract weather event type, city, threshold."""

    TEMP_PATTERNS = [
        ("reach", "above"), ("exceed", "above"), ("hit", "above"),
        ("above", "above"), ("over", "above"), ("high of", "above"),
        ("below", "below"), ("drop below", "below"), ("under", "below"),
        ("low of", "below"),
    ]
    RAIN_KEYWORDS  = ["rain", "rainfall", "precipitation", "precip", "inches of rain"]
    SNOW_KEYWORDS  = ["snow", "snowfall", "inches of snow", "blizzard"]
    WIND_KEYWORDS  = ["wind", "winds", "gusts", "mph"]
    STORM_KEYWORDS = ["thunderstorm", "tornado", "hurricane", "severe storm"]

    def classify(self, title: str) -> str:
        t = title.lower()
        if any(k in t for k in self.STORM_KEYWORDS): return "storm"
        if any(k in t for k in self.SNOW_KEYWORDS):  return "snow"
        if any(k in t for k in self.RAIN_KEYWORDS):  return "rain"
        if any(k in t for k in self.WIND_KEYWORDS):  return "wind"
        if "°" in t or "degree" in t or "temp" in t: return "temp"
        if "°f" in t or "fahrenheit" in t:           return "temp"
        return "weather"

    def extract_city(self, title: str) -> str:
        for city in CITY_GRID_MAP:
            if city.lower() in title.lower():
                return city
        # Try common abbreviations
        abbr = {"nyc": "New York", "la": "Los Angeles", "sf": "San Francisco",
                "kc": "Kansas City", "nola": "New Orleans"}
        t = title.lower()
        for k, v in abbr.items():
            if k in t:
                return v
        return "Unknown"

    def extract_threshold(self, title: str, category: str) -> Optional[float]:
        """Try to extract the numeric threshold from the market title."""
        import re
        if category == "temp":
            m = re.search(r'(\d+)\s*°?[Ff]', title)
            if m:
                return float(m.group(1))
        elif category in ("rain", "snow"):
            m = re.search(r'(\d+\.?\d*)\s*inch', title)
            if m:
                return float(m.group(1))
        elif category == "wind":
            m = re.search(r'(\d+)\s*mph', title)
            if m:
                return float(m.group(1))
        return None

    def extract_yes_prob_from_orderbook(self, market_data: dict) -> float:
        """
        Convert Kalshi market data to an implied YES probability.
        Kalshi yes_ask / yes_bid are in cents (0-100).
        Mid-market price = (bid + ask) / 2 / 100
        """
        try:
            yes_bid = market_data.get("yes_bid", 0) or 0
            yes_ask = market_data.get("yes_ask", 100) or 100
            mid = (yes_bid + yes_ask) / 2 / 100.0
            return max(0.01, min(0.99, mid))
        except (TypeError, ZeroDivisionError):
            return 0.5


# ─────────────────────────────────────────────
# Edge Calculator
# ─────────────────────────────────────────────

class EdgeCalculator:
    def __init__(self):
        self.noaa   = NOAAClient()
        self.parser = MarketParser()

    def analyze(self, market_data: dict) -> Optional[WeatherMarket]:
        """
        Take raw Kalshi market data, fetch NOAA data, and compute edge.
        Returns a WeatherMarket object or None if we can't analyze it.
        """
        title    = market_data.get("title", "")
        ticker   = market_data.get("ticker", "")
        category = self.parser.classify(title)
        city     = self.parser.extract_city(title)
        threshold = self.parser.extract_threshold(title, category)
        kalshi_prob = self.parser.extract_yes_prob_from_orderbook(market_data)
        volume   = market_data.get("volume", 0) or 0
        oi       = market_data.get("open_interest", 0) or 0
        res_time = market_data.get("close_time") or market_data.get("expiration_time") or ""

        if city == "Unknown":
            return None

        # Fetch NOAA forecast
        noaa_prob, noaa_source, reasoning = self._compute_noaa_prob(
            city, category, threshold, title
        )

        if noaa_prob is None:
            return None

        edge     = noaa_prob - kalshi_prob
        net_edge = abs(edge)
        direction = "YES" if edge > 0 else "NO"

        if net_edge < MIN_EDGE:
            return None

        return WeatherMarket(
            ticker       = ticker,
            title        = title,
            category     = category,
            city         = city,
            kalshi_yes_prob = kalshi_prob,
            noaa_prob    = noaa_prob,
            edge         = edge,
            direction    = direction,
            net_edge     = net_edge,
            resolution_dt = res_time,
            volume       = volume,
            open_interest = oi,
            noaa_source  = noaa_source,
            reasoning    = reasoning,
        )

    def _compute_noaa_prob(self, city: str, category: str,
                            threshold: Optional[float], title: str):
        """Compute NOAA-based probability for a given weather event."""
        forecast = self.noaa.get_forecast(city)
        if not forecast:
            return None, None, "No NOAA data available"

        if category == "temp":
            return self._temp_prob(city, threshold, title, forecast)
        elif category == "rain":
            return self._rain_prob(city, threshold, title, forecast)
        elif category == "snow":
            return self._snow_prob(city, threshold, title, forecast)
        elif category == "wind":
            return self._wind_prob(city, threshold, title, forecast)
        elif category == "storm":
            return self._storm_prob(city, title, forecast)
        else:
            return None, None, "Unknown category"

    def _temp_prob(self, city, threshold, title, forecast):
        t_lower = title.lower()
        is_above = any(k in t_lower for k in ["reach", "exceed", "hit", "above", "over", "high of"])
        direction = "above" if is_above else "below"
        forecast_temp = self.noaa.extract_max_temp(forecast) if is_above else self.noaa.extract_min_temp(forecast)

        if forecast_temp is None or threshold is None:
            return 0.5, "NWS Hourly", "Could not extract temp — using neutral prior"

        prob = self.noaa.temp_threshold_probability(forecast_temp, threshold, direction)
        reasoning = (
            f"NWS forecast {direction}-threshold temp: {forecast_temp:.1f}°F vs "
            f"market threshold {threshold:.0f}°F. "
            f"P({direction} {threshold:.0f}°F) = {prob*100:.1f}% using ±4°F forecast uncertainty."
        )
        return prob, "NWS 7-Day Forecast", reasoning

    def _rain_prob(self, city, threshold, title, forecast):
        noaa_precip_prob = self.noaa.extract_precip_probability(forecast)
        forecast_temp = self.noaa.extract_max_temp(forecast) or 50

        if noaa_precip_prob is None:
            return 0.5, "NWS PoP", "No precipitation probability in NWS data"

        # If there's a specific amount threshold, adjust
        if threshold is not None and threshold > 0:
            # Rough heuristic: if PoP is high AND threshold is low, boost probability
            # If threshold is high (e.g. 2+ inches), dampen probability
            amount_factor = max(0.3, 1.0 - (threshold * 0.15))
            prob = noaa_precip_prob * amount_factor
        else:
            prob = noaa_precip_prob

        prob = max(0.05, min(0.95, prob))
        reasoning = (
            f"NWS probability of precipitation: {noaa_precip_prob*100:.0f}%. "
            f"{'Amount-adjusted for ' + str(threshold) + '\" threshold.' if threshold else ''} "
            f"Final NOAA-derived P(rain event) = {prob*100:.1f}%."
        )
        return prob, "NWS PoP (Probability of Precipitation)", reasoning

    def _snow_prob(self, city, threshold, title, forecast):
        forecast_temp = self.noaa.extract_max_temp(forecast)
        noaa_precip_prob = self.noaa.extract_precip_probability(forecast)

        # Snow requires below-freezing temps
        if forecast_temp is not None and forecast_temp > 40:
            prob = 0.05
            reasoning = f"NWS max temp {forecast_temp:.1f}°F — too warm for significant snow. P(snow) = 5%."
        elif noaa_precip_prob is not None:
            # Temperature-gated snow probability
            temp_factor = 1.0 if (forecast_temp or 32) <= 32 else 0.5
            prob = noaa_precip_prob * temp_factor * (0.8 if threshold and threshold > 2 else 1.0)
            prob = max(0.03, min(0.92, prob))
            reasoning = (
                f"NWS PoP {noaa_precip_prob*100:.0f}% with forecast temp {forecast_temp or 'unknown'}°F. "
                f"Snow probability estimate: {prob*100:.1f}%."
            )
        else:
            return None, None, "Insufficient NOAA data for snow probability"

        return prob, "NWS Forecast (temp + PoP)", reasoning

    def _wind_prob(self, city, threshold, title, forecast):
        forecast_wind = self.noaa.extract_max_wind(forecast)

        if forecast_wind is None or threshold is None:
            return 0.5, "NWS Wind", "Could not extract wind forecast"

        prob = self.noaa.wind_threshold_probability(forecast_wind, threshold)
        reasoning = (
            f"NWS forecast max wind: {forecast_wind:.0f} mph vs threshold {threshold:.0f} mph. "
            f"P(wind > {threshold:.0f} mph) = {prob*100:.1f}% using ±5 mph uncertainty."
        )
        return prob, "NWS 7-Day Forecast (Wind)", reasoning

    def _storm_prob(self, city, title, forecast):
        noaa_precip_prob = self.noaa.extract_precip_probability(forecast)
        # Thunderstorms require instability — rough heuristic using PoP + temp
        forecast_temp = self.noaa.extract_max_temp(forecast) or 60

        if noaa_precip_prob is None:
            return 0.5, "NWS PoP", "No storm data"

        # Storms more likely when warm + high precip probability
        temp_boost = max(0.5, min(1.5, (forecast_temp - 50) / 40 + 0.7))
        prob = noaa_precip_prob * 0.4 * temp_boost  # Thunderstorm is subset of all precip
        prob = max(0.03, min(0.85, prob))
        reasoning = (
            f"NWS PoP {noaa_precip_prob*100:.0f}% + max temp {forecast_temp:.0f}°F → "
            f"convective storm estimate: {prob*100:.1f}%. "
            f"(Thunderstorms ≈ 30-40% of high-PoP events at this temperature.)"
        )
        return prob, "NWS PoP + Convective Heuristic", reasoning


# ─────────────────────────────────────────────
# Demo Data (when not authenticated)
# ─────────────────────────────────────────────

def get_demo_markets() -> list[dict]:
    """Return realistic-looking demo markets for testing without credentials."""
    now = datetime.now(timezone.utc)
    markets = []
    examples = [
        ("WEATHER-LV-TEMP-95-0326", "Will Las Vegas reach 95°F on March 26?",
         31, 47, "Las Vegas"),
        ("WEATHER-NYC-RAIN-1IN-0325", "Will NYC get >1 inch rain in next 48hrs?",
         58, 72, "New York"),
        ("WEATHER-CHI-TEMP-20-0324", "Will Chicago temp drop below 20°F tonight?",
         44, 29, "Chicago"),
        ("WEATHER-MIA-RAIN-05-0327", "Will Miami see >0.5in rain on Mar 27?",
         35, 51, "Miami"),
        ("WEATHER-DEN-SNOW-3IN-0329", "Will Denver get >3 inches snow this week?",
         22, 38, "Denver"),
        ("WEATHER-PHX-TEMP-100-0328", "Will Phoenix exceed 100°F on Mar 28?",
         18, 12, "Phoenix"),
        ("WEATHER-BOS-WIND-40-0326", "Will Boston see wind >40mph on Mar 26?",
         29, 44, "Boston"),
        ("WEATHER-HOU-STORM-0327", "Will Houston see a thunderstorm on Mar 27?",
         52, 68, "Houston"),
        ("WEATHER-SEA-RAIN-1IN-0329", "Will Seattle get <1 inch rain this week?",
         41, 27, "Seattle"),
        ("WEATHER-ATL-SNOW-0326", "Will Atlanta see ice/sleet on Mar 26?",
         33, 18, "Atlanta"),
        ("WEATHER-KC-WIND-35-0324", "Will Kansas City wind exceed 35mph today?",
         37, 53, "Kansas City"),
        ("WEATHER-MIN-TEMP-50-0328", "Will Minneapolis hit 50°F this week?",
         44, 61, "Minneapolis"),
    ]
    for ticker, title, yes_bid, noaa_pct, city in examples:
        markets.append({
            "ticker":         ticker,
            "title":          title,
            "yes_bid":        yes_bid,
            "yes_ask":        yes_bid + 4,
            "volume":         int(2000 + abs(yes_bid - noaa_pct) * 200),
            "open_interest":  int(500 + abs(yes_bid - noaa_pct) * 100),
            "close_time":     (now + timedelta(days=3)).isoformat(),
            "_noaa_override": noaa_pct / 100.0,  # Pre-computed for demo
        })
    return markets


# ─────────────────────────────────────────────
# Output Helpers
# ─────────────────────────────────────────────

def cprint(msg: str, color: str = "white"):
    if not COLORS:
        print(msg)
        return
    c = {"green": Fore.GREEN, "red": Fore.RED, "yellow": Fore.YELLOW,
         "cyan": Fore.CYAN, "white": "", "blue": Fore.BLUE,
         "magenta": Fore.MAGENTA}.get(color, "")
    print(f"{c}{msg}{Style.RESET_ALL if c else ''}")

def edge_color(net_edge: float) -> str:
    if net_edge >= STRONG_EDGE: return "green"
    if net_edge >= MEDIUM_EDGE: return "yellow"
    return "white"

def edge_label(net_edge: float) -> str:
    if net_edge >= STRONG_EDGE: return "★ STRONG"
    if net_edge >= MEDIUM_EDGE: return "◆ MEDIUM"
    return "· WEAK  "

def direction_color(direction: str) -> str:
    return "green" if direction == "YES" else "red"

def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║          KALSHI WEATHER EDGE SCANNER  v1.0                    ║
║          NOAA/NWS Integration  |  Las Vegas, NV               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    cprint(banner, "cyan")

def print_market(m: WeatherMarket, idx: int, verbose: bool = False):
    edge_str    = f"+{m.net_edge*100:.1f}%"
    kalshi_str  = f"{m.kalshi_yes_prob*100:.0f}%"
    noaa_str    = f"{m.noaa_prob*100:.0f}%"
    label       = edge_label(m.net_edge)
    color       = edge_color(m.net_edge)
    dir_color   = direction_color(m.direction)

    cprint(f"\n  [{idx:02d}] {m.title[:65]}", "white")
    cprint(f"       City: {m.city:<18} Category: {m.category.upper()}", "white")

    if COLORS:
        kalshi_display = f"{Fore.RED}{kalshi_str}{Style.RESET_ALL}"
        noaa_display   = f"{Fore.GREEN}{noaa_str}{Style.RESET_ALL}"
        edge_display   = f"{Fore.GREEN if m.edge > 0 else Fore.RED}{edge_str}{Style.RESET_ALL}"
        dir_display    = f"{Fore.GREEN if m.direction == 'YES' else Fore.RED}BET {m.direction}{Style.RESET_ALL}"
        label_display  = f"{Fore.GREEN if m.net_edge >= STRONG_EDGE else Fore.YELLOW}{label}{Style.RESET_ALL}"
    else:
        kalshi_display = kalshi_str
        noaa_display   = noaa_str
        edge_display   = edge_str
        dir_display    = f"BET {m.direction}"
        label_display  = label

    print(f"       Kalshi: {kalshi_display}  |  NOAA: {noaa_display}  |  Edge: {edge_display}  |  {label_display}  |  {dir_display}")
    print(f"       Volume: ${m.volume:,}  |  Source: {m.noaa_source}")

    if verbose:
        cprint(f"\n       📡 Analysis: {m.reasoning}", "cyan")

def print_queue(queue: BetQueue):
    cprint("\n" + "═"*65, "cyan")
    cprint("  🎯  SEQUENTIAL BET QUEUE", "cyan")
    cprint("═"*65, "cyan")

    for i, bet in enumerate(queue.bets, 1):
        dir_color = "green" if bet.direction == "YES" else "red"
        cprint(f"  {i}. [{bet.direction}] {bet.title[:55]}", dir_color)
        print(f"     Edge: +{bet.net_edge*100:.1f}%  |  NOAA: {bet.noaa_prob*100:.0f}%  |  Kalshi: {bet.kalshi_yes_prob*100:.0f}%")

    print()
    capital = queue.start_capital
    cprint(f"  Starting capital : ${capital:,.2f}", "white")
    cprint(f"  Bets queued      : {len(queue.bets)} / 9", "white")
    payout = queue.projected_payout()
    win_prob = queue.win_probability()
    cprint(f"  Theoretical max  : ${payout:,.2f} (if all {len(queue.bets)} win at 2:1)", "green")
    cprint(f"  Compound win prob: {win_prob*100:.3f}% (using NOAA estimates)", "yellow")
    cprint(f"  Expected value   : ${payout * win_prob:,.2f}", "yellow")
    cprint("═"*65 + "\n", "cyan")


# ─────────────────────────────────────────────
# Main Scanner
# ─────────────────────────────────────────────

class WeatherScanner:
    def __init__(self):
        self.kalshi    = KalshiClient()
        self.calc      = EdgeCalculator()
        self.parser    = MarketParser()
        self.demo_mode = False

    def run(self, args):
        print_banner()
        cprint("  Connecting to Kalshi...", "white")
        authenticated = self.kalshi.login()

        if not authenticated:
            self.demo_mode = True
            cprint("  Running in DEMO MODE — no real trades will be placed.\n", "yellow")

        cprint("  Fetching weather markets...", "white")
        raw_markets = self._fetch_markets()
        cprint(f"  Analyzing {len(raw_markets)} markets against NOAA data...\n", "white")

        opportunities = []
        for i, raw in enumerate(raw_markets):
            sys.stdout.write(f"\r  Scanning market {i+1}/{len(raw_markets)}...")
            sys.stdout.flush()

            if self.demo_mode and "_noaa_override" in raw:
                # Use pre-computed demo probabilities
                opp = self._demo_analyze(raw)
            else:
                opp = self.calc.analyze(raw)

            if opp:
                opportunities.append(opp)

        print()  # newline after progress

        # Filter
        if args.city:
            opportunities = [o for o in opportunities if args.city.lower() in o.city.lower()]
        if args.category:
            opportunities = [o for o in opportunities if o.category == args.category]
        opportunities = [o for o in opportunities if o.net_edge >= args.min_edge]

        # Sort by edge
        opportunities.sort(key=lambda x: x.net_edge, reverse=True)

        if not opportunities:
            cprint("\n  No opportunities found matching your filters.\n", "yellow")
            return

        # Display
        top_n = args.top or len(opportunities)
        cprint(f"\n  Found {len(opportunities)} opportunities. Showing top {min(top_n, len(opportunities))}:\n", "green")
        cprint("  " + "─"*63, "white")

        for i, opp in enumerate(opportunities[:top_n], 1):
            print_market(opp, i, verbose=args.verbose)

        # Queue
        if args.auto_queue or args.top == 9:
            queue = BetQueue(start_capital=args.capital)
            for opp in opportunities[:9]:
                queue.add(opp)
            print_queue(queue)

            if args.execute and not self.demo_mode:
                self._execute_queue(queue)
            elif args.execute and self.demo_mode:
                cprint("  ⚠  Cannot execute orders in demo mode. Set credentials in .env\n", "yellow")

    def _fetch_markets(self) -> list[dict]:
        if self.demo_mode:
            return get_demo_markets()
        return self.kalshi.get_weather_markets()

    def _demo_analyze(self, raw: dict) -> Optional[WeatherMarket]:
        """Analyze a demo market using pre-computed probabilities."""
        title    = raw.get("title", "")
        ticker   = raw.get("ticker", "")
        category = self.parser.classify(title)
        city     = self.parser.extract_city(title)
        kalshi_prob = self.parser.extract_yes_prob_from_orderbook(raw)
        noaa_prob   = raw.get("_noaa_override", 0.5)
        volume   = raw.get("volume", 0)
        oi       = raw.get("open_interest", 0)
        res_time = raw.get("close_time", "")

        edge      = noaa_prob - kalshi_prob
        net_edge  = abs(edge)
        direction = "YES" if edge > 0 else "NO"

        if net_edge < MIN_EDGE:
            return None

        reasoning = (
            f"NOAA model probability ({noaa_prob*100:.0f}%) diverges from "
            f"Kalshi crowd ({kalshi_prob*100:.0f}%) by {net_edge*100:.1f}%. "
            f"Recommended: BET {direction}."
        )

        return WeatherMarket(
            ticker        = ticker,
            title         = title,
            category      = category,
            city          = city,
            kalshi_yes_prob = kalshi_prob,
            noaa_prob     = noaa_prob,
            edge          = edge,
            direction     = direction,
            net_edge      = net_edge,
            resolution_dt = res_time,
            volume        = volume,
            open_interest = oi,
            noaa_source   = "NOAA Demo Data",
            reasoning     = reasoning,
        )

    def _execute_queue(self, queue: BetQueue):
        """Place orders for all queued bets on Kalshi."""
        cprint("  ⚡ EXECUTING QUEUE...\n", "yellow")
        capital_per_bet = queue.start_capital  # All-in sequential strategy

        for i, bet in enumerate(queue.bets, 1):
            contracts = max(1, int(capital_per_bet / (100 * bet.kalshi_yes_prob + 1)))
            price_cents = int(bet.kalshi_yes_prob * 100) if bet.direction == "YES" \
                         else int((1 - bet.kalshi_yes_prob) * 100)
            side  = bet.direction.lower()

            cprint(f"  [{i}] Placing order: {bet.ticker} | {side.upper()} | "
                   f"{contracts} contracts @ {price_cents}¢", "cyan")

            result = self.kalshi.place_order(
                ticker   = bet.ticker,
                side     = side,
                count    = contracts,
                price    = price_cents
            )

            if result:
                cprint(f"       ✓ Order placed: {result.get('order', {}).get('order_id', 'OK')}", "green")
            else:
                cprint(f"       ✗ Order failed — stopping queue", "red")
                break
            time.sleep(0.5)  # Rate limiting


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi Weather Edge Scanner — find mispriced weather markets using NOAA data"
    )
    parser.add_argument("--city",       type=str,   default=None,
                        help="Filter by city name (e.g. 'Las Vegas')")
    parser.add_argument("--category",   type=str,   default=None,
                        choices=["temp", "rain", "snow", "wind", "storm"],
                        help="Filter by weather category")
    parser.add_argument("--min-edge",   type=float, default=MIN_EDGE,
                        help=f"Minimum edge to show (default: {MIN_EDGE})")
    parser.add_argument("--top",        type=int,   default=None,
                        help="Show top N opportunities")
    parser.add_argument("--auto-queue", action="store_true",
                        help="Auto-build optimal 9-bet sequential queue")
    parser.add_argument("--execute",    action="store_true",
                        help="Execute orders on Kalshi (requires credentials)")
    parser.add_argument("--capital",    type=float, default=200.0,
                        help="Starting capital in dollars (default: $200)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Show detailed NOAA reasoning for each market")
    parser.add_argument("--demo",       action="store_true",
                        help="Force demo mode (no real API calls)")

    args = parser.parse_args()

    scanner = WeatherScanner()
    if args.demo:
        scanner.demo_mode = True

    try:
        scanner.run(args)
    except KeyboardInterrupt:
        cprint("\n\n  Scan interrupted.\n", "yellow")


if __name__ == "__main__":
    main()
