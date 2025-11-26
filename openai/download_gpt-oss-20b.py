from huggingface_hub import snapshot_download
snapshot_download(
repo_id="openai/gpt-oss-20b",
    local_dir="/models/gpt-oss-20b",
    resume_download=True,   # resumes partial data
    max_workers=1,          # lower RAM spikes
    )
