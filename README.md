[![Moondream](https://img.shields.io/badge/Moon-dream-2E3436)](https://moondream.ai/)
[![Replicate](https://img.shields.io/badge/Replicate-Demo_&_Cloud_API-blue)](https://replicate.com/)

# vikhyat/Moondream Cog model

https://replicate.com/jyoung105/moondream

This is an implementation of [vikhyat/Moondream](https://github.com/vikhyat/Moondream) as a [Cog](https://github.com/replicate/cog) model.

## License

The model is release for research purposes only, commercial use is not allowed.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction

```bash
    cog predict -i image=@example0.png -i prompt="Explain it in one sentence."
