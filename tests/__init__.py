import pathlib
from typing import List


class TestData:
    synth_seg_path = pathlib.Path(__file__).parent.parent.resolve()

    @classmethod
    def get_label_maps(cls) -> List[str]:
        """
        Provides the absolute paths of the training label maps.
        Returns:
            List of all training label map paths
        """
        label_dir = TestData.synth_seg_path / "data" / "training_label_maps"
        return list(map(lambda p: p.absolute().as_posix(), label_dir.glob("*.nii.gz")))
