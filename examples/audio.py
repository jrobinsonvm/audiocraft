import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio


processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["90 Hiphop with loud and pounding drums", "Mixed with 2000s R&B track song with swinging guitar and loud high pitch brass instruments"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=512)

sampling_rate = model.config.audio_encoder.sampling_rate

scipy.io.wavfile.write("musicgen_out-1.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
