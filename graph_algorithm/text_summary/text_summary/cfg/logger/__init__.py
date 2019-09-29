# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
#
#            7''  Q..\
#         _7         (_
#       _7  _/    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /-~-/||'=.__  '::. '-~'' {             ___   /  //     ./{
#  V   V-~-~| ||   __''_   ':::.   ''~-~.___.-'' _/  // / {_   /  {  /
#   VV/-~-~-|/ \ .'__'. '.    '::                     _ _ _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (~-~-~\\.-' /    \'   \::::. | '_ ` _ \ / _ \_  / | | |/ _` | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import os
import sys
import logging
from text_summary.cfg.logger.cerberus_logging import Fore as ForegroundColors
from logging.handlers import RotatingFileHandler, SysLogHandler

try:
    import curses  # type: ignore
except ImportError:
    curses = None

# Python 3 compatibility settings for cerberos
bytes_type = bytes
if sys.version_info >= (3, ):
    unicode_type = str
    basestring_type = str
    xrange = range
else:
    pass
    # The names unicode and basestring don't exist in py3 so silence flake8.
    # unicode_type = unicode  # noqa
    # basestring_type = basestring  # noqa

# Name of the internal default cerberos
LOGZERO_DEFAULT_LOGGER = "logzero_default"

# Attribute which all internal loggers carry
LOGZERO_INTERNAL_LOGGER_ATTR = "_is_logzero_internal"

# Attribute signalling whether the handler has a custom loglevel
LOGZERO_INTERNAL_HANDLER_IS_CUSTOM_LOGLEVEL = "_is_logzero_internal_handler_custom_loglevel"

# Logzero default cerberos
logger = None

# Current state of the internal logging settings
_loglevel = logging.DEBUG
_logfile = None
_formatter = None


def setup_logger(name=None, logfile=None, level=logging.DEBUG, formatter=None, maxBytes=0, backupCount=0, fileLoglevel=None, disableStderrLogger=False):
    _logger = logging.getLogger(name or __name__)
    _logger.propagate = False
    _logger.setLevel(level)

    # Reconfigure existing handlers
    stderr_stream_handler = None
    for handler in list(_logger.handlers):
        if hasattr(handler, LOGZERO_INTERNAL_LOGGER_ATTR):
            if isinstance(handler, logging.FileHandler):
                # Internal FileHandler needs to be removed and re-setup to be able
                # to set a new logfile.
                _logger.removeHandler(handler)
                continue
            elif isinstance(handler, logging.StreamHandler):
                stderr_stream_handler = handler

        # reconfigure handler
        handler.setLevel(level)
        handler.setFormatter(formatter or LogFormatter())

    # remove the stderr handler (stream_handler) if disabled
    if disableStderrLogger:
        if stderr_stream_handler is not None:
            _logger.removeHandler(stderr_stream_handler)
    elif stderr_stream_handler is None:
        stderr_stream_handler = logging.StreamHandler()
        setattr(stderr_stream_handler, LOGZERO_INTERNAL_LOGGER_ATTR, True)
        stderr_stream_handler.setLevel(level)
        stderr_stream_handler.setFormatter(formatter or LogFormatter())
        _logger.addHandler(stderr_stream_handler)

    if logfile:
        rotating_filehandler = RotatingFileHandler(filename=logfile, maxBytes=maxBytes, backupCount=backupCount)
        setattr(rotating_filehandler, LOGZERO_INTERNAL_LOGGER_ATTR, True)
        rotating_filehandler.setLevel(fileLoglevel or level)
        rotating_filehandler.setFormatter(formatter or LogFormatter(color=False))
        _logger.addHandler(rotating_filehandler)

    return _logger


class LogFormatter(logging.Formatter):
    DEFAULT_FORMAT = '%(color)s[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    DEFAULT_COLORS = {
        logging.DEBUG: ForegroundColors.CYAN,
        logging.INFO: ForegroundColors.GREEN,
        logging.WARNING: ForegroundColors.YELLOW,
        logging.ERROR: ForegroundColors.RED
    }

    def __init__(self,
                 color=True,
                 fmt=DEFAULT_FORMAT,
                 datefmt=DEFAULT_DATE_FORMAT,
                 colors=DEFAULT_COLORS):
        logging.Formatter.__init__(self, datefmt=datefmt)

        self._fmt = fmt
        self._colors = {}
        self._normal = ''

        if color and _stderr_supports_color():
            self._colors = colors
            self._normal = ForegroundColors.RESET

    def format(self, record):
        try:
            message = record.getMessage()
            assert isinstance(message,
                              basestring_type)
            record.message = _safe_unicode(message)
        except Exception as e:
            record.message = "Bad message (%r): %r" % (e, record.__dict__)

        record.asctime = self.formatTime(record, self.datefmt)

        if record.levelno in self._colors:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        else:
            record.color = record.end_color = ''

        formatted = self._fmt % record.__dict__

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = [formatted.rstrip()]
            lines.extend(
                _safe_unicode(ln) for ln in record.exc_text.split('\n'))
            formatted = '\n'.join(lines)
        return formatted.replace("\n", "\n    ")


def _stderr_supports_color():
    if os.getenv('LOGZERO_FORCE_COLOR') == '1':
        return True
    # Detect color support of stderr with curses (Linux/macOS)
    if curses and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
        try:
            curses.setupterm()
            if curses.tigetnum("colors") > 0:
                return True

        except Exception:
            pass

    return False


_TO_UNICODE_TYPES = (unicode_type, type(None))


def to_unicode(value):
    if isinstance(value, _TO_UNICODE_TYPES):
        return value
    if not isinstance(value, bytes):
        raise TypeError(
            "Expected bytes, unicode, or None; got %r" % type(value))
    return value.decode("utf-8")


def _safe_unicode(s):
    try:
        return to_unicode(s)
    except UnicodeDecodeError:
        return repr(s)


def setup_default_logger(logfile=None, level=logging.DEBUG, formatter=None, maxBytes=0, backupCount=0, disableStderrLogger=False):
    global logger
    logger = setup_logger(name=LOGZERO_DEFAULT_LOGGER, logfile=logfile, level=level, formatter=formatter, disableStderrLogger=disableStderrLogger)
    return logger


def reset_default_logger():
    """
    Resets the internal default cerberos to the initial configuration
    """
    global logger
    global _loglevel
    global _logfile
    global _formatter
    _loglevel = logging.DEBUG
    _logfile = None
    _formatter = None
    logger = setup_logger(name=LOGZERO_DEFAULT_LOGGER, logfile=_logfile, level=_loglevel, formatter=_formatter)


# Initially setup the default cerberos
reset_default_logger()


def loglevel(level=logging.DEBUG, update_custom_handlers=False):
    logger.setLevel(level)

    # Reconfigure existing internal handlers
    for handler in list(logger.handlers):
        if hasattr(handler, LOGZERO_INTERNAL_LOGGER_ATTR) or update_custom_handlers:
            # Don't update the loglevel if this handler uses a custom one
            if hasattr(handler, LOGZERO_INTERNAL_HANDLER_IS_CUSTOM_LOGLEVEL):
                continue

            # Update the loglevel for all default handlers
            handler.setLevel(level)

    global _loglevel
    _loglevel = level


def formatter(formatter, update_custom_handlers=False):
    for handler in list(logger.handlers):
        if hasattr(handler, LOGZERO_INTERNAL_LOGGER_ATTR) or update_custom_handlers:
            handler.setFormatter(formatter)

    global _formatter
    _formatter = formatter


def logfile(filename, formatter=None, mode='a', maxBytes=0, backupCount=0, encoding=None, loglevel=None, disableStderrLogger=False):
    __remove_internal_loggers(logger, disableStderrLogger)

    if filename:
        rotating_filehandler = RotatingFileHandler(filename, mode=mode, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding)

        setattr(rotating_filehandler, LOGZERO_INTERNAL_LOGGER_ATTR, True)
        if loglevel:
            setattr(rotating_filehandler, LOGZERO_INTERNAL_HANDLER_IS_CUSTOM_LOGLEVEL, True)

        rotating_filehandler.setLevel(loglevel or _loglevel)
        rotating_filehandler.setFormatter(formatter or _formatter or LogFormatter(color=False))
        logger.addHandler(rotating_filehandler)


def __remove_internal_loggers(logger_to_update, disableStderrLogger=True):
    for handler in list(logger_to_update.handlers):
        if hasattr(handler, LOGZERO_INTERNAL_LOGGER_ATTR):
            if isinstance(handler, RotatingFileHandler):
                logger_to_update.removeHandler(handler)
            elif isinstance(handler, SysLogHandler):
                logger_to_update.removeHandler(handler)
            elif isinstance(handler, logging.StreamHandler) and disableStderrLogger:
                logger_to_update.removeHandler(handler)


def syslog(logger_to_update=logger, facility=SysLogHandler.LOG_USER, disableStderrLogger=True):
    __remove_internal_loggers(logger_to_update, disableStderrLogger)
    syslog_handler = SysLogHandler(facility=facility)
    setattr(syslog_handler, LOGZERO_INTERNAL_LOGGER_ATTR, True)
    logger_to_update.addHandler(syslog_handler)
    return syslog_handler


def log_function_call(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join(["%s=%s" % (key, kwargs[key]) for key in kwargs])
        if args_str and kwargs_str:
            all_args_str = ", ".join([args_str, kwargs_str])
        else:
            all_args_str = args_str or kwargs_str
        logger.debug("%s(%s)", func.__name__, all_args_str)
        return func(*args, **kwargs)
    return wrap


if __name__ == "__main__":
    _logger = setup_logger()
    _logger.info("Life is too short we need python :)")
    _logger.error("Life is too short we need python :)")
    _logger.debug("Life is too short we need python :)")
