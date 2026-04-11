import pytest

pytestmark = pytest.mark.skip(reason="Manual script, not a test")

if __name__ == "__main__":
    print("PDF test_contract.pdf is already generated. The old PDF library was removed.")
