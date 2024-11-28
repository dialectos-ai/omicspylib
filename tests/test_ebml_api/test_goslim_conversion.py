import os

import pytest
from omicspylib import go_to_goslim


@pytest.mark.skipif(
    not os.getenv('RUN_EMBL_API_CALLS', False),
    reason='Set RUN_EMBL_API_CALLS env variable to True to run.')
def test_go_to_goslim():
    # setup
    from_ids = ['GO:0000922', 'GO:0035253']
    to_ids = ['GO:0005929', 'GO:0005856']
    expected_output = {
        'GO:0035253': ['GO:0005929', 'GO:0005856'],
        'GO:0000922': ['GO:0005856']
    }

    # action
    go_mapping = go_to_goslim(from_ids, to_ids)

    # assertion
    for key, value in go_mapping.items():
        for v in value:
            assert v in expected_output[key]
