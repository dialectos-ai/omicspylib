import time
from typing import List

import requests


def go_to_goslim(from_ids: List[str], to_ids: List[str],
                 relations: str ='is_a,part_of,occurs_in,regulates',
                 batch_size: int = 100, delay_time: float = 0.1) -> dict:
    """
    Map a list of GO term ids, to the corresponding GO-slim terms.

    You might use this function to provide a lookup table
    of the raw GO terms, e.g., as extracted from Uniprot, to a fixed
    "vocabulary" of GO-slim terms, of your interest.

    Once you have this mapping, you can rename the GO terms of your
    original dataset.

    There is no guarantee for 1-to-1 mapping.
    Both ``from_ids`` and ``to_ids`` will be reduced to their unique
    values internally.

    Parameters
    ----------
    from_ids: list
        A list of GO ids, e.g. from your proteins as loaded from Uniprot.
    to_ids: list
        A list of GO ids from your slim set. E.g. a generic slim set
        of your interest.
    relations: str
        The set of relationships over which the slimming information is
        computed, joined with comma.
        Defaults to ``is_a,part_of,occurs_in,regulates`` (see API doc in notes).
    batch_size: int
        Requests to EMBL API are made with batches of the specified size.
        Then, results are merged into one response. Too large batches
        can cause 400 errors.
    delay_time: float
        Delay time in seconds between consecutive requests.

    Returns
    -------
    list
        A list of matching raw GO terms with GO-slims.

    Notes
    -----
    For more information about the endpoint, visit the API documentation at:
    https://www.ebi.ac.uk/QuickGO/api/index.html#!/gene_ontology/findSlimsUsingGET

    Examples
    --------
    >>> from omicspylib import go_to_goslim
    >>> from_ids = ['GO:0000922','GO:0035253']
    >>> to_ids = ['GO:0005929', 'GO:0005856']
    >>> go_mapping = go_to_goslim(from_ids, to_ids)  # doctest: +SKIP
    >>> print(go_mapping)
    # Expected output:
    {'GO:0035253': ['GO:0005929', 'GO:0005856'],
    'GO:0000922': ['GO:0005856']}
    """
    url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/slim'
    from_ids = list(set(from_ids))
    to_ids = list(set(to_ids))
    from_ids = [go_id.strip() for go_id in from_ids]
    to_ids = [go_id.strip() for go_id in to_ids]

    n_items = len(from_ids)
    n_steps = len(from_ids) // batch_size + 1

    all_mappings = []
    for i in range(n_steps):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_items)

        params = {
            'slimsToIds': ','.join(to_ids),
            'slimsFromIds': ','.join(from_ids[start_idx:end_idx]),
            'relations': relations
        }
        response = requests.get(url, params=params)
        time.sleep(delay_time)
        if response.ok:
            batched_mappings = response.json()['results']
            all_mappings.extend(batched_mappings)
        else:
            response.raise_for_status()

    # reformat outputs
    return {r['slimsFromId']: r['slimsToIds'] for r in all_mappings}
