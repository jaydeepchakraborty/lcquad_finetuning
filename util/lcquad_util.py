from lcquad_finetuning.util.util_lib import *

class LCQuadUtil:

    @staticmethod
    def get_curr_tm():
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("YR-%Y_MM-%m_DD-%d_HR-%H_M-%M_SEC-%S")  # e.g., 2025_10_21_20_05_00
        return timestamp_str

    @staticmethod
    def log_mps_memory(logger, tag=""):
        if torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
            driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
            logger.info(f"[MPS Memory {tag}] allocated: {allocated:.2f} GB | driver_allocated: {driver:.2f} GB")
        else:
            logger.info(f"[MPS Memory {tag}] MPS not available")


