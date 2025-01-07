from typing import List, Literal, Tuple, Type
from pathlib import Path
import skimage.io
import numpy as np
from pydantic import BaseModel, Field, validator
import uvicorn
import imaging_server_kit as serverkit
import rembg


class Parameters(BaseModel):
    image: str = Field(
        title="Image",
        description="Input image (2D grayscale, RGB)",
        json_schema_extra={"widget_type": "image"},
    )
    rembg_model_name: Literal["silueta", "isnet", "u2net", "u2netp", "sam"] = Field(
        default="silueta",
        title="Model",
        description="The model used for background removal.",
        json_schema_extra={"widget_type": "dropdown"},
    )

    @validator("image", pre=False, always=True)
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array


class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str = "rembg",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

        self.sessions: dict[str, rembg.sessions.BaseSession] = {}

    def run_algorithm(
        self, image: np.ndarray, rembg_model_name: str = "silueta", **kwargs
    ) -> List[Tuple]:
        """Binary segmentation using rembg."""

        session = self.sessions.setdefault(
            rembg_model_name, rembg.new_session(rembg_model_name)
        )

        if rembg_model_name == "sam":
            x0, y0, x1, y1 = 0, 0, image.shape[0], image.shape[1]

            prompt = [
                {
                    "type": "rectangle",
                    "data": [y0, x0, y1, x1],
                    "label": 2,  # `label` is irrelevant for SAM in bounding boxes mode
                }
            ]

            segmentation = rembg.remove(
                data=image,
                session=session,
                only_mask=True,
                post_process_mask=True,
                sam_prompt=prompt,
                **kwargs,
            )
            segmentation = segmentation == 0  # Invert it (for some reason)

        else:
            segmentation = rembg.remove(
                data=image,
                session=session,
                only_mask=True,
                post_process_mask=True,
                **kwargs,
            )
            segmentation = segmentation == 255

        segmentation_params = {
            "name": f"{rembg_model_name}_result",
        }

        return [
            (segmentation, segmentation_params, "labels"),
        ]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = Server()
app = server.app


if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
