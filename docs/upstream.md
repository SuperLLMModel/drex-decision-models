# Kev attribution and provenance

The Python runtime incorporates code from [Kev](https://github.com/jaredpalmer/kev),
Copyright 2026 Jared Palmer, under the Apache License, Version 2.0.
The copyright notice is retained in [NOTICE](../NOTICE), and the license text
is in [LICENSE](../LICENSE).

The following files have been adapted by Nace.AI for Drex:

| File | Drex adaptations |
|---|---|
| `kev/api.py` | Drex model default and ordered score legends. |
| `kev/model.py` | Efficient-DLM loading, bidirectional state attention, and configurable serving limits. |
| `kev/checkpoint.py` | Simplified loading of local merged Drex checkpoints and pointer heads. |

## Revision record

- **Original upstream revision:** unknown. This repository's initial import
  (`84d611cc31f50fc1e8815a400e0637bdffb73c32`) did not record the Kev commit used.
- **Comparison reference:**
  [`fc4a17e194bca35a48c57211392ec49ea54a94ac`](https://github.com/jaredpalmer/kev/tree/fc4a17e194bca35a48c57211392ec49ea54a94ac),
  inspected on October 9, 2026. This pins the source and
  [license](https://github.com/jaredpalmer/kev/blob/fc4a17e194bca35a48c57211392ec49ea54a94ac/LICENSE)
  used for the attribution review; it is not a claim that this was the original
  import revision. Differences from this reference may include later upstream
  changes as well as Drex changes.

For future imports, record the full upstream commit SHA, the imported paths,
and a summary of local modifications here. If the original revision is recovered,
update the original-revision entry with the evidence identifying it.
