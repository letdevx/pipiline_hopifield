# Agent Rules & Directives

## Jupyter Notebook Editing via Jupytext

To prevent file corruption and formatting errors associated with direct manipulation of `.ipynb` files:

1. **NEVER EDIT `.ipynb` DIRECTLY**: Never edit JSON structure in `.ipynb` files directly (neither manually nor via scripts).
2. **VERIFY & PAIR WITH JUPYTEXT**: Whenever requested to create or edit a Jupyter Notebook (`.ipynb`):
   - Check if a paired `.py` file exists.
   - If not paired, pair it using Jupytext:
     ```bash
     jupytext --set-formats ipynb,py:percent <notebook>.ipynb
     ```
3. **EDIT THE `.py` SCRIPT**: Perform all code edits on the paired `.py` script (percent format).
4. **SYNCHRONIZE NOTEBOOK**: Immediately after modifying the `.py` script, execute synchronization to update the `.ipynb` file:
   ```bash
   jupytext --sync <notebook>.py
   ```
