import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from app.chunking.chunk_schema import CodeChunk


class GitHubReferenceBuilder:
    """
    Build GitHub URLs for repository chunks.

    Example:

    src/components/model_trainer.py
    lines 31-86

    becomes:

    https://github.com/pvats2006/ML-Project/blob/main/
    src/components/model_trainer.py#L31-L86
    """

    def __init__(
        self,
        repo_path: str
    ):

        self.repo_path = Path(
            repo_path
        )

        self.remote_url = (
            self.get_remote_url()
        )

        self.branch = (
            self.get_current_branch()
        )

        self.owner = None
        self.repository = None

        self.parse_remote_url()

    # ==================================================
    # GET GIT REMOTE
    # ==================================================

    def get_remote_url(self):

        try:

            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_path),
                    "remote",
                    "get-url",
                    "origin",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout.strip()

        except subprocess.CalledProcessError:

            return None

    # ==================================================
    # GET CURRENT BRANCH
    # ==================================================

    def get_current_branch(self):

        try:

            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_path),
                    "branch",
                    "--show-current",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            branch = result.stdout.strip()

            return branch or "main"

        except subprocess.CalledProcessError:

            return "main"

    # ==================================================
    # PARSE GITHUB URL
    # ==================================================

    def parse_remote_url(self):

        if not self.remote_url:

            return

        remote = (
            self.remote_url.strip()
        )

        # ----------------------------------------------
        # HTTPS
        #
        # https://github.com/user/repo.git
        # ----------------------------------------------

        if remote.startswith(
            "https://"
        ):

            parsed = urlparse(
                remote
            )

            if parsed.hostname != "github.com":

                return

            path = (
                parsed.path
                .strip("/")
            )

        # ----------------------------------------------
        # SSH
        #
        # git@github.com:user/repo.git
        # ----------------------------------------------

        elif remote.startswith(
            "git@github.com:"
        ):

            path = (
                remote
                .replace(
                    "git@github.com:",
                    "",
                    1
                )
                .strip("/")
            )

        else:

            return

        # Remove .git
        path = re.sub(
            r"\.git$",
            "",
            path
        )

        parts = path.split(
            "/"
        )

        if len(parts) != 2:

            return

        self.owner = parts[0]

        self.repository = parts[1]

    # ==================================================
    # BUILD GITHUB URL
    # ==================================================

    def build_url(
        self,
        file_path: str,
        line_start: int,
        line_end: int
    ):

        if not self.owner:
            return None

        if not self.repository:
            return None

        # ----------------------------------------------
        # Normalize path
        # ----------------------------------------------

        clean_path = (
            file_path
            .replace("\\", "/")
            .lstrip("/")
        )

        # ----------------------------------------------
        # Encode spaces safely
        # ----------------------------------------------

        clean_path = clean_path.replace(
            " ",
            "%20"
        )

        # ----------------------------------------------
        # Single line
        # ----------------------------------------------

        if line_start == line_end:

            line_reference = (
                f"#L{line_start}"
            )

        # ----------------------------------------------
        # Line range
        # ----------------------------------------------

        else:

            line_reference = (
                f"#L{line_start}-L{line_end}"
            )

        return (
            f"https://github.com/"
            f"{self.owner}/"
            f"{self.repository}/"
            f"blob/"
            f"{self.branch}/"
            f"{clean_path}"
            f"{line_reference}"
        )

    # ==================================================
    # BUILD REFERENCE
    # ==================================================

    def build_reference(
        self,
        chunk: CodeChunk
    ):

        github_url = self.build_url(
            file_path=chunk.file_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
        )

        if not github_url:

            chunk.metadata.update({
                "github_reference": None,
                "github_reference_available": False,
            })

            return chunk

        # ----------------------------------------------
        # Add metadata
        # ----------------------------------------------

        chunk.metadata.update({

            "github_reference": github_url,

            "github_reference_available": True,

            "github_owner": self.owner,

            "github_repository":
                self.repository,

            "github_branch":
                self.branch,

            "github_file":
                chunk.file_path,

            "github_line_start":
                chunk.line_start,

            "github_line_end":
                chunk.line_end,

        })

        return chunk

    # ==================================================
    # BUILD REFERENCES FOR MANY CHUNKS
    # ==================================================

    def build_references(
        self,
        chunks
    ):

        enriched_chunks = []

        for chunk in chunks:

            enriched_chunks.append(
                self.build_reference(
                    chunk
                )
            )

        return enriched_chunks


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    file_path = input(
        "Enter file path: "
    ).strip()

    line_start = int(
        input(
            "Enter line start: "
        ).strip()
    )

    line_end = int(
        input(
            "Enter line end: "
        ).strip()
    )

    builder = GitHubReferenceBuilder(
        repo_path
    )

    print(
        "\nGIT INFORMATION\n"
    )

    print(
        "Remote:",
        builder.remote_url
    )

    print(
        "Owner:",
        builder.owner
    )

    print(
        "Repository:",
        builder.repository
    )

    print(
        "Branch:",
        builder.branch
    )

    url = builder.build_url(
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
    )

    print(
        "\nGITHUB REFERENCE\n"
    )

    print(url)