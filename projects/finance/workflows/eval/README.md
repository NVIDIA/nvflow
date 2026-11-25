# Finance evaluation configs

These files define standalone and embedded NeMo-Gym evaluation campaigns.

For supported files, model presets, benchmark selection, checkpoint conversion,
commands, outputs, and result interpretation, see the project-owned
[evaluation workflow guide](../../docs/workflows/05-eval.md).

The hosted judge in `base.yaml` points at the NVIDIA inference hub and needs an
API key in the job environment. To use OpenAI, or to see how each backend
delivers the key, read [Hosted Model Endpoints](../../../../docs/hosted-endpoints.md).
