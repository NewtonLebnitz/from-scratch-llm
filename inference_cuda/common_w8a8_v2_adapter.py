"""Process-local isolated W8A8 correction, no frozen file or model mutation."""
import common_calibration_v1 as calibration
from common_w8a8_v2 import W8A8Linear
from common_aux_v1_adapter import Engine,configure_model
# Only this new worker imports this adapter. The original auxiliary process and
# its frozen files keep the original M=16 probe/results intact.
calibration.W8A8Linear=W8A8Linear
