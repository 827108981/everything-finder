"""Extend the frozen vector experiment with downstream traces for every arm."""
from pathlib import Path
from tools import validate_vector_packaging_20261116 as experiment


def main():
    original_write = experiment.write

    def preserve_original_audit(path, data):
        path = Path(path)
        if path.name == 'native_audit.json':
            path = path.with_name('native_audit_all_stages_20261117.json')
        return original_write(path, data)

    experiment.write = preserve_original_audit
    experiment.downstream(['W1', 'W2', 'W3', 'W4', 'W5', 'F10_count_matched'])


if __name__ == '__main__':
    main()
