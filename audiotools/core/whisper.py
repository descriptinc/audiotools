import torch
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor


class WhisperMixin:
    is_initialized = False

    def setup_whisper(
        self,
        pretrained_model_name_or_path: str = "openai/whisper-base.en",
        device: str = "cuda",
    ):
        self.whisper_device = device
        self.whisper_processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.whisper_device)
        self.is_initialized = True

    def get_whisper_features(self):
        if not self.is_initialized:
            self.setup_whisper()

        signal = self.to(self.device)
        raw_speech = [
            s[0]
            for s in signal.resample(
                self.whisper_processor.feature_extractor.sampling_rate
            ).numpy()
        ]

        with torch.inference_mode():
            input_features = self.whisper_processor(
                raw_speech,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).input_features

        return input_features

    def get_whisper_transcript(self) -> str:
        if not self.is_initialized:
            self.setup()

        input_features = self.get_whisper_features()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            generated_ids = self.whisper_model.generate(inputs=input_features)

        transcription = self.whisper_processor.batch_decode(generated_ids)
        return transcription

    def get_whisper_embeddings(self) -> torch.Tensor:
        if not self.is_initialized:
            self.setup()

        input_features = self.get_whisper_features()
        encoder = self.whisper_model.get_encoder()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            embeddings = encoder(input_features)

        return embeddings.last_hidden_state
