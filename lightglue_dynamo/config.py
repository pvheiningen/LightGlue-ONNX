from enum import StrEnum, auto


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()
    tensorrt = auto()


class Extractor(StrEnum):
    superpoint = auto()
    # disk = auto()

    @property
    def dim(self) -> int:
        match self:
            case Extractor.superpoint:
                return 256
            # case Extractor.disk:
            #     return 128
