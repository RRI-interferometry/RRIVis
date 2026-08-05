"""Tests for the Rich console helpers in :mod:`radiosim.utils.logging`.

The four ``print_*`` helpers accept caller-supplied messages that are plain
text, not Rich markup. Bracketed text such as a model-name list
(``[gsm2008, haslam]``) must survive to the rendered output verbatim instead
of being parsed (and silently eaten) as a markup tag. These are the API-002
regression tests: they capture the console output and assert the literal
message survives each helper, including a message shaped exactly like the
``Simulator.setup()`` offline pre-flight warning
(``src/radiosim/api/simulator.py:782``).
"""

import logging

from radiosim.utils.logging import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)


def _capture(func, message: str) -> str:
    """Run one helper under console capture and return whitespace-normalized text.

    Normalizing collapses Rich's line wrapping so substring assertions do not
    depend on the capture console's width.
    """
    with console.capture() as capture:
        func(message)
    return " ".join(capture.get().split())


class TestPrintHelpersRenderBracketsLiterally:
    """A bracketed model-name list must survive every print_* helper."""

    def test_print_warning_keeps_bracketed_model_list(self):
        out = _capture(print_warning, "Sky model(s) [gsm2008] unavailable")
        assert "[gsm2008]" in out

    def test_print_error_keeps_bracketed_model_list(self):
        out = _capture(print_error, "Loader [haslam] failed")
        assert "[haslam]" in out

    def test_print_success_keeps_bracketed_model_list(self):
        out = _capture(print_success, "Loaded [gsm2016] successfully")
        assert "[gsm2016]" in out

    def test_print_info_keeps_bracketed_model_list(self):
        out = _capture(print_info, "Selected models: [lfsm, haslam]")
        assert "[lfsm, haslam]" in out

    def test_glyph_prefix_survives_alongside_literal_message(self):
        out = _capture(print_warning, "plain [bracketed] text")
        assert out.startswith("⚠")
        assert "plain [bracketed] text" in out


class TestOfflinePreflightMessageShape:
    """The exact message shape from api/simulator.py:782 must print intact."""

    def test_offline_preflight_warning_keeps_model_names(self):
        # Mirror the f-string construction at the call site.
        display = "pygdsm data"
        model_names = ", ".join(["gsm2008", "haslam"])
        message = (
            f"Sky model(s) [{model_names}] require {display} but network is unavailable"
        )
        out = _capture(print_warning, message)
        assert (
            "Sky model(s) [gsm2008, haslam] require pygdsm data "
            "but network is unavailable" in out
        )


class TestRichHandlerLiteralMessages:
    """logger.* messages routed through the RichHandler print literally too."""

    def test_logged_message_keeps_bracketed_text(self):
        logger = logging.getLogger("radiosim")
        saved_handlers = list(logger.handlers)
        saved_level = logger.level
        try:
            setup_logging(level=logging.INFO)
            with console.capture() as capture:
                logger.warning("Sky model(s) [gsm2008] require network")
            out = " ".join(capture.get().split())
            assert "[gsm2008]" in out
        finally:
            logger.handlers = saved_handlers
            logger.setLevel(saved_level)
