"""A voice on this machine, for when there is no key or no connection.

Windows SAPI is tried first (powershell.exe and the voices Windows ships,
so nothing is installed), then espeak-ng if it is on PATH. With neither the
provider reports itself unavailable and the service moves on to a silent
track.

The text and paths reach PowerShell through environment variables, never
spliced into the command line, so a line with quotes or a `$` in it cannot
change what runs. Quality is a plain system voice: this keeps the pipeline
finishing with sound, it is not the voice for a finished video.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from src.models.speech.base import Audio, SpeechError, SpeechProvider, wav_duration

logger = logging.getLogger(__name__)

TIMEOUT_S = 90
SAPI_RATE = 1            # SAPI runs -10..10; one notch up from default suits a promotional read
ESPEAK_WPM = 165

_SAPI_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $v = $env:XT_VOICE
  try {
    if ($v -eq 'female') { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) }
    elseif ($v -eq 'male') { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Male) }
    elseif ($v) { $s.SelectVoice($v) }
  } catch { }
  $s.Rate = [int]$env:XT_RATE
  $s.SetOutputToWaveFile($env:XT_OUT)
  $s.Speak($env:XT_TEXT)
} finally { $s.Dispose() }
"""


class OfflineSpeechProvider(SpeechProvider):
    name = "offline"

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 which: Callable[[str], Optional[str]] = shutil.which, platform: str = sys.platform):
        self._run = runner               # injectable for tests
        self._which = which
        self._platform = platform

    def _powershell(self) -> Optional[str]:
        if not self._platform.startswith("win"):
            return None
        return self._which("powershell.exe") or self._which("powershell")

    def _espeak(self) -> Optional[str]:
        return self._which("espeak-ng") or self._which("espeak")

    def backend(self) -> str:
        """"sapi", "espeak-ng", or "" when this machine has no voice to use."""
        if self._powershell():
            return "sapi"
        return "espeak-ng" if self._espeak() else ""

    def is_available(self) -> bool:
        return bool(self.backend())

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "no speech engine on this machine (needs Windows SAPI or espeak-ng)"

    # -- engines ----------------------------------------------------------------

    def _sapi(self, text: str, voice: Optional[str], out: Path) -> None:
        env = {**os.environ, "XT_TEXT": text, "XT_VOICE": voice or "", "XT_OUT": str(out),
               "XT_RATE": str(SAPI_RATE)}
        self._exec([self._powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-Command", _SAPI_SCRIPT], env=env)

    def _espeak_ng(self, text: str, voice: Optional[str], out: Path) -> None:
        name = {"female": "en+f3", "male": "en+m3"}.get(voice or "", voice or "en")
        self._exec([self._espeak(), "-v", name, "-s", str(ESPEAK_WPM), "-w", str(out), "--stdin"], stdin=text)

    def _exec(self, args: list, env: Optional[dict] = None, stdin: Optional[str] = None) -> None:
        try:
            result = self._run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=TIMEOUT_S, env=env, input=stdin)
        except subprocess.TimeoutExpired as exc:
            raise SpeechError(f"the local voice did not finish in {TIMEOUT_S}s") from exc
        except OSError as exc:
            raise SpeechError(f"the local voice could not be started: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-300:] or f"exit code {result.returncode}"
            raise SpeechError(f"the local voice failed: {detail}")

    # -- the provider contract ------------------------------------------------------

    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        backend = self.backend()
        if not backend:
            raise SpeechError(self.unavailable_reason())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        (self._sapi if backend == "sapi" else self._espeak_ng)(text, voice, output_path)
        try:
            duration = wav_duration(output_path)
        except Exception as exc:  # noqa: BLE001 - a missing or unreadable file is the same answer
            raise SpeechError(f"the local voice ({backend}) wrote no readable audio") from exc
        if duration <= 0:
            raise SpeechError(f"the local voice ({backend}) wrote an empty file")
        return Audio(path=output_path, duration_s=duration, provider=self.name, voice=voice or "",
                     model_id=backend)
