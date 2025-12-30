from lcquad_finetuning.util.util_lib import *

class LCQuadUtil:

    @staticmethod
    def get_curr_tm():
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("YR-%Y_MM-%m_DD-%d_HR-%H_M-%M_SEC-%S")  # e.g., 2025_10_21_20_05_00
        print(f"running for {timestamp_str}")
        return timestamp_str


