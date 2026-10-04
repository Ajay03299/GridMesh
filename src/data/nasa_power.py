"""Auditable hourly Indian-weather adapter; no interpolated sub-hourly observations.

NASA POWER GHI and temperature are gridded estimates, NOT measured PV. Solar
position uses a low-order astronomical approximation. Haurwitz clear-sky GHI
is a deterministic geometry-only reference, NOT tomorrow's observed weather.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_power(path, latitude=12.9716, longitude=77.5946):
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    parameters = obj["properties"]["parameter"]
    if obj["parameters"]["ALLSKY_SFC_SW_DWN"]["units"] != "Wh/m^2":
        raise ValueError("Expected hourly irradiation in Wh/m^2")
    if obj["parameters"]["T2M"]["units"] != "C":
        raise ValueError("Expected temperature in Celsius")
    df = pd.DataFrame(parameters).sort_index()
    if set(df.columns) != {"ALLSKY_SFC_SW_DWN", "T2M"}:
        raise ValueError("Expected archived GHI and temperature parameters")
    if (df == obj["header"]["fill_value"]).any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("NASA record contains missing/nonfinite values; explicit repair required")
    utc = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    if len(utc) != 8784 or not ((utc[1:] - utc[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected a complete hourly leap-year record (8784 observations)")
    if utc[0] != pd.Timestamp("2024-01-01", tz="UTC") or utc[-1] != pd.Timestamp("2024-12-31 23:00", tz="UTC"):
        raise ValueError("Unexpected record dates")
    coordinates = obj["geometry"]["coordinates"]
    if abs(coordinates[0] - longitude) > .02 or abs(coordinates[1] - latitude) > .02:
        raise ValueError("Record coordinates do not match the configured location")
    local = utc.tz_convert("Asia/Kolkata").tz_localize(None)
    doy = utc.dayofyear.to_numpy()
    gamma = 2 * np.pi / 365 * (doy - 1)
    equation = 229.18 * (.000075 + .001868*np.cos(gamma) - .032077*np.sin(gamma)
                         - .014615*np.cos(2*gamma) - .040849*np.sin(2*gamma))
    declination = (.006918 - .399912*np.cos(gamma) + .070257*np.sin(gamma)
                   - .006758*np.cos(2*gamma) + .000907*np.sin(2*gamma)
                   - .002697*np.cos(3*gamma) + .00148*np.sin(3*gamma))
    hour_angle = np.radians((utc.hour.to_numpy()*60 + 4*longitude + equation)/4 - 180)
    lat = np.radians(latitude)
    cosine = np.clip(np.sin(lat)*np.sin(declination) + np.cos(lat)*np.cos(declination)*np.cos(hour_angle), -1, 1)
    cs = np.zeros(len(df));sun = cosine > 0
    cs[sun] = 1098 * cosine[sun] * np.exp(-.059/cosine[sun])
    # Wh/m2 over a one-hour bin / 1 hour = bin-average W/m2 (same numerical value).
    frame = pd.DataFrame({"timestamp":local,"GHI":df.ALLSKY_SFC_SW_DWN.to_numpy() / 1.0,
        "Temperature":df.T2M.to_numpy(),"Clearsky GHI":cs,
        "Solar Zenith Angle":np.degrees(np.arccos(cosine))})
    provenance = {"provider":"NASA POWER","latitude":latitude,"longitude":longitude,
        "year_utc":2024,"rows":len(frame),"step_hours":1,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_header":obj["header"],"pv_status":"modeled from gridded GHI and temperature, not measured generation",
        "clear_sky_status":"derived Haurwitz + approximate solar geometry",
        "ghi_conversion":"hourly Wh/m2 divided by one hour to obtain average W/m2",
        "weather_diversity":"one co-located weather record; sites are simulated",
        "source_url":"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude=77.5946&latitude=12.9716&start=20240101&end=20241231&format=JSON&time-standard=UTC"}
    return frame, provenance
