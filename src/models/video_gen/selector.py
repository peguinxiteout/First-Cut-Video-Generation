"""The model selector: which providers exist, which can run right now, what
they would cost, and which chain a choice turns into.

Read from configs/video/providers.yaml and from the machine, never from a
list in the UI, so the Video tab, the CLI and GET /video/providers show the
same thing. Availability is worked out when asked (at page load and again
before a run), not remembered.

A provider is available when
    its adapter is registered         (else "adapter not installed")
    its key resolves                  (else "no key: set FAL_KEY")
    no quota marker is active         (else "allowance spent ...")

Chains always end with pan-and-zoom:
    auto        [first available by auto_priority, kenburns]
    a provider  [that provider, kenburns]
    kenburns    [kenburns]      never calls a model
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.core.config import load_yaml_config
from src.core.exceptions import ConfigError
from src.models.video_gen import registry
from src.models.video_gen.base import ProviderSpec

logger = logging.getLogger(__name__)

CATALOG_FILE = "video/providers.yaml"
AUTO = "auto"
PAN_AND_ZOOM = "kenburns"


@dataclass(frozen=True)
class ProviderOption:
    id: str
    label: str
    available: bool
    reason: str                          # why not, or (for auto) what it will use
    est_cost_usd: Optional[float]        # None = price not set, or credit-based (see cost_basis)
    auto_priority: Optional[int] = None
    vendor: str = ""
    cost_basis: str = ""                 # per_second | per_clip | credits
    price_from: bool = False             # the estimate is the lowest tier
    free_tier: str = ""
    watermark_free_tier: bool = False
    verified: bool = True                # False: the price was given to us and not checked


@dataclass(frozen=True)
class ChainPlan:
    ids: tuple[str, ...]                 # providers in the order they are tried
    note: str = ""                       # set when the choice could not be honoured
    est_cost_usd: Optional[float] = None


def _settings(settings):
    if settings is not None:
        return settings
    from src.core.config import settings as _default

    return _default


def load_catalog() -> dict[str, ProviderSpec]:
    raw = load_yaml_config(CATALOG_FILE).get("providers") or {}
    catalog: dict[str, ProviderSpec] = {}
    for provider_id, entry in raw.items():
        entry = entry or {}
        catalog[provider_id] = ProviderSpec(
            id=provider_id, label=entry.get("label", provider_id), adapter=entry["adapter"],
            key_env=entry.get("key_env") or None, auto_priority=entry.get("auto_priority"),
            cost_per_second=entry.get("cost_per_second"), max_duration_s=float(entry.get("max_duration_s", 8)),
            supports_first_last_frame=bool(entry.get("supports_first_last_frame", False)),
            hidden=bool(entry.get("hidden", False)), endpoint=str(entry.get("endpoint") or ""),
            params=dict(entry.get("params") or {}), vendor=str(entry.get("vendor") or ""),
            model_id=str(entry.get("model_id") or ""), cost_basis=str(entry.get("cost_basis") or ""),
            cost_per_clip=entry.get("cost_per_clip"), clip_s=entry.get("clip_s"),
            cost_by_resolution=dict(entry.get("cost_by_resolution") or {}),
            price_from=bool(entry.get("price_from", False)), free_tier=str(entry.get("free_tier") or ""),
            watermark_free_tier=bool(entry.get("watermark_free_tier", False)),
            verified=bool(entry.get("verified", True)), price_source=str(entry.get("price_source") or ""))
    if PAN_AND_ZOOM not in catalog:
        raise ConfigError(f"{CATALOG_FILE} must define '{PAN_AND_ZOOM}': it is the last link of every chain")
    return catalog


def resolve_key(key_env: Optional[str], settings=None) -> str:
    """The key named by `key_env`: a Settings field of that name (which reads
    .env) or the process environment, as the model gateway resolves its models."""
    if not key_env:
        return ""
    value = getattr(_settings(settings), key_env.lower(), None) or os.getenv(key_env) or ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Quota markers: a provider that said its allowance is spent is skipped for a
# while, by the next run as well, rather than rediscovered one failed shot at a time.
# ---------------------------------------------------------------------------

def _marker_path(settings=None) -> Path:
    return _settings(settings).data_dir_resolved / "video_cache" / "quota.json"


def _read_markers(settings=None) -> dict[str, str]:
    try:
        data = json.loads(_marker_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def quota_until(provider_id: str, settings=None, now: Optional[datetime] = None) -> Optional[datetime]:
    """When the provider may be tried again, or None if it is not held back."""
    now = now or datetime.now(timezone.utc)
    try:
        until = datetime.fromisoformat(_read_markers(settings).get(provider_id, ""))
    except ValueError:
        return None
    return until if until > now else None


def mark_quota(provider_id: str, settings=None, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    hours = float(load_yaml_config(CATALOG_FILE).get("quota_cooldown_hours", 12))
    until = now + timedelta(hours=hours)
    markers = {k: v for k, v in _read_markers(settings).items() if quota_until(k, settings, now)}
    markers[provider_id] = until.isoformat()
    path = _marker_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(markers), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not record the quota marker for %s: %s", provider_id, exc)
    return until


# ---------------------------------------------------------------------------
# Availability, options, chains
# ---------------------------------------------------------------------------

def availability(spec: ProviderSpec, settings=None, now: Optional[datetime] = None) -> tuple[bool, str]:
    if spec.adapter not in registry.ADAPTERS:
        return False, "adapter not installed"
    if spec.key_env and not resolve_key(spec.key_env, settings):
        return False, f"no key: set {spec.key_env}"
    until = quota_until(spec.id, settings, now)
    if until is not None:
        return False, f"allowance spent; skipped until {until.strftime('%H:%M UTC')}"
    return True, ""


def _estimate(spec: ProviderSpec, duration_s: float) -> Optional[float]:
    rate = spec.rate_per_second
    return None if rate is None else round(rate * duration_s, 4)


def _auto_pick(catalog: dict[str, ProviderSpec], settings, now) -> Optional[ProviderSpec]:
    candidates = [s for s in catalog.values()
                  if s.auto_priority is not None and availability(s, settings, now)[0]]
    return min(candidates, key=lambda s: s.auto_priority) if candidates else None


def options(duration_s: float = 15, settings=None, now: Optional[datetime] = None) -> list[ProviderOption]:
    """Everything the selector offers: Auto, each provider, pan-and-zoom last.
    Unavailable ones are listed with the reason, not left out."""
    catalog = load_catalog()
    pick = _auto_pick(catalog, settings, now)
    out = [ProviderOption(
        id=AUTO, label="Auto", available=True,
        reason=f"Uses {pick.label}" if pick else "No video model is available; uses pan-and-zoom",
        est_cost_usd=_estimate(pick, duration_s) if pick else 0.0)]
    for spec in catalog.values():
        if spec.hidden or spec.id == PAN_AND_ZOOM:
            continue
        ok, reason = availability(spec, settings, now)
        out.append(ProviderOption(id=spec.id, label=spec.label, available=ok, reason=reason,
                                  est_cost_usd=_estimate(spec, duration_s), auto_priority=spec.auto_priority,
                                  vendor=spec.vendor, cost_basis=spec.cost_basis, price_from=spec.price_from,
                                  free_tier=spec.free_tier, watermark_free_tier=spec.watermark_free_tier,
                                  verified=spec.verified))
    out.append(ProviderOption(id=PAN_AND_ZOOM, label=catalog[PAN_AND_ZOOM].label, available=True, reason="",
                              est_cost_usd=0.0))
    return out


def resolve(choice: Optional[str], duration_s: float = 15, settings=None,
            now: Optional[datetime] = None) -> ChainPlan:
    """The chain a choice turns into. A named provider that cannot run now
    gives the pan-and-zoom chain with a note saying why, not an error: the
    video is still made. Raises ConfigError for an id that does not exist."""
    catalog = load_catalog()
    choice = (choice or _settings(settings).video_gen_provider or AUTO).strip().lower()
    if choice == AUTO:
        pick = _auto_pick(catalog, settings, now)
        if pick is None:
            return ChainPlan((PAN_AND_ZOOM,), est_cost_usd=0.0)
        return ChainPlan((pick.id, PAN_AND_ZOOM), est_cost_usd=_estimate(pick, duration_s))
    if choice == PAN_AND_ZOOM:
        return ChainPlan((PAN_AND_ZOOM,), est_cost_usd=0.0)
    spec = catalog.get(choice)
    if spec is None:
        raise ConfigError(f"Unknown video provider '{choice}'. Choose auto or one of: "
                          f"{', '.join(i for i, s in catalog.items() if not s.hidden)}")
    ok, reason = availability(spec, settings, now)
    if not ok:
        return ChainPlan((PAN_AND_ZOOM,), note=f"{spec.label} is unavailable ({reason}); using pan-and-zoom",
                         est_cost_usd=0.0)
    return ChainPlan((spec.id, PAN_AND_ZOOM), est_cost_usd=_estimate(spec, duration_s))
