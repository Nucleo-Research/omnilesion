"""Copy the OmniLesion trainer into the installed nnU-Net v2 package.

nnU-Net resolves trainers by class name by scanning ``nnunetv2/training/nnUNetTrainer``; there is no plugin mechanism,
so the module has to live inside that directory. Run once after ``pip install nnunetv2==2.6.0``::

    python -m omnilesion.nnunet_trainer.install

and check with::

    python -m omnilesion.nnunet_trainer.install --check
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

MODULE = "nnUNetTrainerOmniLesion"


def target_dir() -> Path:
    import nnunetv2

    return Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only verify that nnU-Net can resolve the trainer")
    arguments = parser.parse_args()
    destination = target_dir() / f"{MODULE}.py"
    if not arguments.check:
        source = Path(__file__).resolve().parent / f"{MODULE}.py"
        shutil.copyfile(source, destination)
        print(f"copied {source.name} -> {destination}")
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    trainer = recursive_find_python_class(str(target_dir()), MODULE, "nnunetv2.training.nnUNetTrainer")
    if trainer is None:
        print(f"nnU-Net could not resolve {MODULE} in {target_dir()}", file=sys.stderr)
        return 1
    module = importlib.import_module(f"nnunetv2.training.nnUNetTrainer.{MODULE}")
    print(f"resolved {trainer} from {module.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
