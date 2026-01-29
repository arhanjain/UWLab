"""Launch RSL training on AWS SageMaker."""

import os
import subprocess
import tempfile
import time
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import boto3
import sagemaker
import tyro
from sagemaker.aws_batch.training_queue import TrainingQueue as Queue
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch

NAME = "uwlab-rsl"
INSTANCE_MAPPER = {
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    # "p5test": "ml.p5.48xlarge",
}
GPU_MAPPER = {
    "ml.p4de.24xlarge": 8,
    "ml.p5.48xlarge": 8,
    "ml.p5en.48xlarge": 8,
    "ml.p4d.24xlarge": 8,
}
QUEUE_MAPPER = {
    "us-west-2": {
        "ml.p5.48xlarge": "fss-ml-p5-48xlarge-us-west-2",
        # "ml.p5.48xlarge": "fss-testing-p5-48xlarge-us-west-2",
        "ml.p5en.48xlarge": "fss-ml-p5en-48xlarge-us-west-2",
        "ml.p4de.24xlarge": "fss-ml-p4de-24xlarge-us-west-2",
        "ml.p4d.24xlarge": "fss-ml-p4d-24xlarge-us-west-2",
    },
}

@dataclass
class RLArgs:
    """Arguments for RL training - will be converted to config file."""
    task: str
    """Task name"""
    num_envs: int = 4096
    """Number of environments"""
    seed: int = 42
    """Random seed"""
    max_iterations: int = 5000
    """Maximum training iterations"""
    logger: str = "wandb"
    """Logger type (wandb, tensorboard, neptune)"""
    experiment_name: str | None = None
    """Experiment name for logging"""
    run_name: str | None = None
    """Run name suffix"""

    extra: str = ""
    """Extra flags to pass to the training script"""

@dataclass
class LaunchRSLTrainingArgs:
    """Arguments for launching RSL training on SageMaker."""

    rl_args: RLArgs
    """Arguments for RL training"""
    # SageMaker configuration
    user: str   # type: ignore
    """Your username for naming"""
    region: str = "us-west-2"
    """AWS region"""
    profile: str = "default"
    """AWS profile"""
    arn: str = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"
    """SageMaker execution role ARN"""
    instance_type: Literal["p4de", "p5", "p5en"] = "p4de"
    """Instance type"""
    instance_count: int = 1
    """Number of instances"""
    max_run_days: int = 2
    """Max run time in days"""
    volume_size: int = 100
    """Volume size in GB"""
    name_prefix: str | None = None
    """Job name prefix"""
    local: bool = False
    """Build container and test locally with Docker (skips SageMaker)"""

    # SageMaker queue args
    queue_name: str = "ml"
    """SageMaker queue name prefix"""
    priority: int = 20
    """Job priority in queue"""


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)

def get_image(user, profile="default", region="us-west-2", local=False):
    """Build and push Docker image to ECR (or build locally for local mode)."""
    docker_dir = Path(__file__).parent.parent  # Go up to repo root
    algorithm_name = f"{user}-{NAME}"
    dockerfile_path = Path(__file__).parent / "Dockerfile.uwlab"

    if local:
        # For local mode, just build the image locally without pushing to ECR
        print("Building container for local testing")
        run_command(f"docker build --progress=plain -f {dockerfile_path} -t {algorithm_name} {docker_dir}")
        return algorithm_name

    # For remote mode, build and push to ECR
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    assert account.isdigit(), f"Invalid account value: {account}"

    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin"
    )

    print("Building container for ECR")
    commands = [
        # Log in to SageMaker account for base images
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        # Log in to NVIDIA NGC if needed for Isaac Sim
        # Build from repo root with dockerfile in sagemaker/
        f"docker build --progress=plain -f {dockerfile_path} --build-arg AWS_REGION={region} -t {algorithm_name} {docker_dir}",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} --no-cli-pager || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name} --no-cli-pager"
        ),
    ]

    # Create command, making sure to exit if any part breaks
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

def load_secrets():
    """Load secrets from secrets.env file."""
    # Load secrets for environment variables
    env_vars = {}
    secrets_file = Path("secrets.env")
    if secrets_file.exists():
        with open(secrets_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    clean_key = key.strip()
                    clean_value = value.strip().strip('"').strip("'")
                    # env_vars.append(f"-e {clean_key}={clean_value}")
                    env_vars[clean_key] = clean_value

    return env_vars


def generate_config_from_args(rl_args: RLArgs) -> str:
    """Generate YAML config file from RLArgs and return path.

    Args:
        rl_args: RL training arguments

    Returns:
        Path to generated config file
    """

    config = {
        "training": {
            "task": rl_args.task,
            "num_envs": rl_args.num_envs,
            "seed": rl_args.seed,
            "max_iterations": rl_args.max_iterations,
            "headless": True,
            "enable_cameras": False,
            "video": False,
            "logger": rl_args.logger,
        },
    }

    # Add optional fields
    # if rl_args.experiment_name:
    #     config["agent"]["experiment_name"] = rl_args.experiment_name
    # if rl_args.run_name:
    #     config["agent"]["run_name"] = rl_args.run_name
    # if rl_args.logger == "wandb" and rl_args.experiment_name:
    #     config["agent"]["wandb_project"] = rl_args.experiment_name

    # Write to temp file
    config_dir = Path("configs/generated")
    config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_dir / f"training_config_{timestamp}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Generated config file: {config_path}")
    return str(config_path)



def main(args: LaunchRSLTrainingArgs | None = None):
    """Launch RSL training on SageMaker."""
    if args is None:
        args = tyro.cli(LaunchRSLTrainingArgs)
    assert args is not None  # Type narrowing for linter

    # Get ARN from environment if not provided
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set SAGEMAKER_ARN"
        args.arn = os.environ["SAGEMAKER_ARN"]

    # Generate config file from RL args
    config_path = generate_config_from_args(args.rl_args)
    print(f"Using config file: {config_path}")

    # Build Docker image (and push to ECR if not local mode)
    image = get_image(args.user, region=args.region, profile=args.profile, local=args.local)

    if args.local:
        # Local mode: run Docker directly for testing
        print("\n=== Local Testing Mode ===")
        print("Building and running container locally without SageMaker\n")

        # Build command line args - just pass config file
        # cmd_args = f"--config {config_path}"
        env_vars = load_secrets()
        env_vars["CONFIG_PATH"] = f"/opt/ml/code/{config_path}"
        env_vars["SAGEMAKER_PROGRAM"] = "scripts/reinforcement_learning/rsl_rl/train.py"
        env_vars["EXTRA_FLAGS"] = f"\"{args.rl_args.extra}\""
        env_vars_str = " ".join([f"-e {key}={value}" for key, value in env_vars.items()])

        # Run docker container
        docker_cmd = (
            f"docker run --rm --gpus all "
            f"-v {os.path.abspath(config_path)}:/opt/ml/code/{config_path} "
            f"{env_vars_str} "
            "-v ~/.aws:/root/.aws:ro "
            f"{image} "
            f"python "
            f"scripts/reinforcement_learning/rsl_rl/train.py"
            f" --config {config_path}"
        )

        print(f"Running: {docker_cmd}\n")
        run_command(docker_cmd)
        print("\n=== Local test completed ===")
        return

    os.environ["AWS_DEFAULT_REGION"] = args.region
    # Create SageMaker session
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    role = args.arn
    role_name = role.split("/")[-1]
    print(f"SageMaker Execution Role: {role}")
    print(f"Role name: {role_name}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account: {account}")

    # Create job name
    def sanitize_name(name):
        name = name.replace("_", "-")
        clean = "".join(c if c.isalnum() or c == "-" else "" for c in name)
        return clean.strip("-") or "job"

    base_job_name = sanitize_name(
        f"{args.name_prefix + '-' if args.name_prefix else ''}{args.user.replace('.', '-')}-{NAME}"
    )

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f"{base_job_name}-{date_str}"[:63].rstrip("-")

    # Pass config file path through hyperparameters
    hyperparameters = {
        "config": config_path
    }

    # Environment variables
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "NCCL_DEBUG": "INFO",
        "ISAAC_HEADLESS": "1",  # Run Isaac Sim in headless mode
        "MY_SCRIPT": "scripts/reinforcement_learning/rsl_rl/train.py",
        "EXTRA_FLAGS": args.rl_args.extra,
        "CONFIG_PATH": f"/opt/ml/code/{config_path}"
    }
    environment.update(load_secrets())

    # Checkpoint path for model saving
    checkpoint_local_path = "/opt/ml/checkpoints"

    # Build torchrun command for distributed training
    instance_type_full = INSTANCE_MAPPER[args.instance_type]
    num_gpus = GPU_MAPPER.get(instance_type_full, 8)


    # estimator = PyTorch(
    #     entry_point="sagemaker/launch.sh",
    #     # entry_point="scripts/reinforcement_learning/rsl_rl/train.py",
    #     # entry_point="torchrun --nnodes=1 --nproc_per_node=8 scripts/reinforcement_learning/rsl_rl/train.py",
    #     image_uri=image,
    #     role=role,
    #     hyperparameters=hyperparameters,
    #     instance_count=args.instance_count,
    #     instance_type=INSTANCE_MAPPER[args.instance_type],
    #     sagemaker_session=sagemaker_session,
    #     base_job_name=base_job_name,
    #     environment=environment,
    #     max_run=args.max_run_days * 24 * 60 * 60,
    #     volume_size=args.volume_size,
    #     checkpoint_local_path=checkpoint_local_path,
    #     input_mode="FastFile",
    #     keep_alive_period_in_seconds=5 * 60,  # Keep instance alive for 5 minutes after job completion
    #     distribution={"torch_distributed": {"enabled": True}},
    #     # distribution=None
    # )


    # # Create generic estimator
    # # Note: We're not using hyperparameters since we build the full command in container_entry_point
    estimator = Estimator(
        entry_point="sagemaker/launch.sh",
        image_uri=image,
        role=role,
        instance_count=args.instance_count,
        instance_type=INSTANCE_MAPPER[args.instance_type],
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        environment=environment,
        max_run=args.max_run_days * 24 * 60 * 60,
        volume_size=args.volume_size,
        checkpoint_local_path=checkpoint_local_path,
        input_mode="FastFile",
        keep_alive_period_in_seconds=5 * 60,  # Keep instance alive for 5 minutes after job completion
        # container_entry_point=container_entry_point,
    )

    print(f"\nStarting training job: {job_name}")
    print(f"Instance type: {INSTANCE_MAPPER[args.instance_type]}")
    print(f"Instance count: {args.instance_count}")
    print(f"GPUs per instance: {num_gpus}")
    print(f"Total GPUs: {args.instance_count * num_gpus}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Environment: {environment}")

    # Submit to training queue
    queue = Queue(
        queue_name=QUEUE_MAPPER[args.region][INSTANCE_MAPPER[args.instance_type]].replace("ml", args.queue_name)
    )
    queue.map(
        estimator,
        inputs=[None],
        job_names=[job_name],
        priority=args.priority,
        share_identifier="default",
        timeout={"attemptDurationSeconds": args.max_run_days * 24 * 60 * 60},
    )
    print(f"\nQueued training job: {job_name}")
    print(f"Priority: {args.priority}")
    # print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}")


if __name__ == "__main__":
    main()
