# NVFlow documentation

NVFlow is the workflow orchestration platform that loads, validates, plans,
packages, and runs independently owned ML workflow projects and their optional
recipe DAGs. Projects build workflows from their own stages and from NVFlow's
shared stages.

Choose the path that matches your role.

## Run an existing workflow

1. [Install NVFlow](../INSTALL.md), then set up a [Slurm](slurm/README.md) or
   [Ray](ray/README.md) target.
2. [Learn the core concepts](getting-started/concepts.md).
3. [Run the starter workflow](getting-started/first-workflow.md).
4. Choose another maintained project:
   - [Finance](../projects/finance/README.md)
   - [HopChain](../projects/hopchain/README.md)
5. [Copy the project starter](../projects/starter/README.md) for a new project.

## Create or extend a project

- [Create an NVFlow project from the starter](getting-started/creating-a-project.md)
- [Project ownership and recipe graph contract](platform/recipe-contract.md)
- [Workflow loading and planning](platform/workflow-and-planning.md)
- [CLI and Python API](platform/cli-and-python-api.md)
- [Build a document-grounded domain project](guides/document-grounded-projects.md)
- [Train with NVFlow's shared stages](guides/training-stages.md)
- [Dedup, decontaminate, and filter datasets](guides/data-stages.md)
- [Generate rubrics with shared stages](guides/rubric-generation.md)

## Operate NVFlow

- [Cluster configuration](operations/cluster-configuration.md)
- [Slurm execution](slurm/README.md)
- [Ray execution](ray/README.md)
- [Remote and air-gapped launch](operations/remote-launch.md)
- [Hosted model endpoints](hosted-endpoints.md)
- [Build and stage the worker images](maintainers/containers.md)
- [NeMo ecosystem integration boundaries](platform/integrations/nemo-ecosystem.md)

## Understand or contribute to the platform

- [Platform architecture](platform/architecture.md)
- [Console UI development](development/console-ui.md)
- [Migrate stages to exact execution](development/migrating-stages-to-execution.md)
- [Ray support validation matrix](development/ray-validation.md)
- [Ray execution assumptions](development/ray-assumptions.md)
- [Architecture diagrams](development/diagrams/README.md)
- [Trace viewer](guides/trace-viewer.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)

Concrete project behavior, models, prompts, expected outputs, and quality claims
belong inside each project. Platform documentation defines only the
contracts NVFlow supports.
