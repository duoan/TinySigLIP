"""
Test script for COCOCaptionDataset with new batch-based teacher cache format.
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import tinysiglip
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from tinysiglip.coco_dataset import collate_coco_batch


def create_mock_cache(cache_dir: Path, num_images: int = 10, num_batches: int = 2):
    """Create mock teacher cache data for testing."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    metadata = {
        "teacher_model_name": "google/siglip2-base-patch16-224",
        "split": "train",
        "max_samples": num_images,
        "dataset_size_name": "tiny",
        "num_images": num_images,
        "total_captions": num_images * 5,  # 5 captions per image
        "num_batches": num_batches,
        "images_per_batch": num_images // num_batches,
        "image_embed_dim": 768,
        "text_embed_dim": 768,
        "cache_dir": str(cache_dir),
        "index_file": str(cache_dir / "image_index.pt"),
    }
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Create batch files, image index, and sample indices
    image_index = {}  # image_id -> (batch_file_idx, local_idx)
    sample_indices = []  # sample_idx -> (image_id, caption_idx)
    images_per_batch = num_images // num_batches

    for batch_idx in range(num_batches):
        batch_data = {}
        start_idx = batch_idx * images_per_batch
        end_idx = min(start_idx + images_per_batch, num_images)

        for local_idx, img_idx in enumerate(range(start_idx, end_idx)):
            # Create mock embeddings
            image_emb = torch.randn(768)
            image_id = img_idx + 1  # COCO IDs start from 1
            image_path = f"train/{image_id:012d}.jpg"

            # Create 5 captions per image
            caption_data_list = []
            for cap_idx in range(5):
                caption_id = img_idx * 5 + cap_idx
                caption_emb = torch.randn(768)
                caption_text = f"Caption {cap_idx} for image {image_id}"
                caption_data_list.append((caption_id, caption_emb, caption_text))
                # Add to sample_indices: (image_id, caption_idx)
                sample_indices.append((image_id, cap_idx))

            # Store: (image_id, image_path, image_emb, caption_data_list)
            batch_data[local_idx] = (image_id, image_path, image_emb, caption_data_list)
            # Use image_id as key (not img_idx) to match actual implementation
            image_index[image_id] = (batch_idx, local_idx)

        # Save batch file
        batch_file = cache_dir / f"batch_{batch_idx:06d}.pt"
        torch.save(batch_data, batch_file)

    # Save image index
    torch.save(image_index, cache_dir / "image_index.pt")

    # Save sample indices
    with open(cache_dir / "sample_indices.pkl", "wb") as f:
        pickle.dump(sample_indices, f)

    return metadata, image_index


def test_cache_loading():
    """Test loading teacher cache."""
    print("Testing cache loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache" / "test_model" / "tiny" / "train"
        metadata, image_index = create_mock_cache(cache_dir, num_images=10, num_batches=2)

        # Try to create dataset with cache (will fail without COCO data, but we can test cache loading)
        # We'll test the cache loading logic directly
        cache_path = cache_dir
        metadata_file = cache_path / "metadata.json"
        index_file = cache_path / "image_index.pt"

        assert metadata_file.exists(), "Metadata file should exist"
        assert index_file.exists(), "Index file should exist"

        # Load metadata
        with open(metadata_file, encoding="utf-8") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["num_images"] == 10, "Metadata should match"
        assert loaded_metadata["num_batches"] == 2, "Metadata should match"

        # Load image index (keyed by image_id, not img_idx)
        loaded_index = torch.load(index_file, map_location="cpu")
        assert len(loaded_index) == 10, "Index should have 10 entries"
        # image_id = 1 (img_idx=0) should be in batch 0, local idx 0
        assert loaded_index[1] == (0, 0), "Image ID 1 should be in batch 0, local idx 0"
        # image_id = 6 (img_idx=5) should be in batch 1, local idx 0
        assert loaded_index[6] == (1, 0), "Image ID 6 should be in batch 1, local idx 0"

        print("✓ Cache loading test passed")


def test_batch_loading():
    """Test lazy loading of batch files."""
    print("Testing batch file loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache" / "test_model" / "tiny" / "train"
        create_mock_cache(cache_dir, num_images=10, num_batches=2)

        # Simulate the batch loading logic
        batch_cache = {}
        batch_file = cache_dir / "batch_000000.pt"
        batch_data = torch.load(batch_file, map_location="cpu")
        batch_cache[0] = batch_data

        assert 0 in batch_cache, "Batch 0 should be cached"
        assert len(batch_data) == 5, "Batch 0 should have 5 images"

        # Check data structure
        local_idx = 0
        image_id, image_path, image_emb, caption_data_list = batch_data[local_idx]
        assert isinstance(image_id, int), "Image ID should be int"
        assert isinstance(image_path, str), "Image path should be str"
        assert image_emb.shape == (768,), "Image embedding should have shape (768,)"
        assert len(caption_data_list) == 5, "Should have 5 captions"

        print("✓ Batch loading test passed")


def test_embedding_retrieval():
    """Test retrieving embeddings from cache."""
    print("Testing embedding retrieval...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache" / "test_model" / "tiny" / "train"
        create_mock_cache(cache_dir, num_images=10, num_batches=2)

        # Load index (keyed by image_id)
        image_index = torch.load(cache_dir / "image_index.pt", map_location="cpu")
        batch_cache = {}

        def load_batch(batch_idx):
            if batch_idx not in batch_cache:
                batch_file = cache_dir / f"batch_{batch_idx:06d}.pt"
                batch_cache[batch_idx] = torch.load(batch_file, map_location="cpu")
            return batch_cache[batch_idx]

        # Test retrieving embeddings for different images (using image_id, not img_idx)
        for image_id in [1, 6, 10]:  # image_id = img_idx + 1
            batch_file_idx, local_idx = image_index[image_id]
            batch_data = load_batch(batch_file_idx)
            cached_image_id, image_path, image_emb, caption_data_list = batch_data[local_idx]

            # Verify image_id matches
            assert cached_image_id == image_id, f"Image ID should match: {cached_image_id} != {image_id}"

            # Test different caption indices
            for cap_idx in [0, 2, 4]:
                caption_id, caption_emb, caption_text = caption_data_list[cap_idx]
                assert image_emb.shape == (768,), "Image embedding shape should be (768,)"
                assert caption_emb.shape == (768,), "Caption embedding shape should be (768,)"
                assert isinstance(caption_text, str), "Caption should be string"

        print("✓ Embedding retrieval test passed")


def test_cache_with_real_data():
    """Test cache with real COCO data if available."""
    print("Testing cache with real COCO data...")

    # Check if real cache exists (relative to project root)
    project_root = Path(__file__).parent.parent
    real_cache_path = project_root / "data" / "coco" / "cache" / "google_siglip2-base-patch16-224" / "tiny" / "train"
    if not real_cache_path.exists():
        print("⚠ Real cache not found, skipping real data test")
        return

    # Load metadata
    metadata_file = real_cache_path / "metadata.json"
    index_file = real_cache_path / "image_index.pt"

    if not metadata_file.exists() or not index_file.exists():
        print("⚠ Cache files not found, skipping real data test")
        return

    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    image_index = torch.load(index_file, map_location="cpu")

    print("  Cache info:")
    print(f"    - Images: {metadata['num_images']}")
    print(f"    - Captions: {metadata['total_captions']}")
    print(f"    - Batches: {metadata['num_batches']}")
    print(f"    - Image embed dim: {metadata['image_embed_dim']}")
    print(f"    - Text embed dim: {metadata['text_embed_dim']}")
    print(f"    - Index entries: {len(image_index)}")

    # Test loading a few batch files
    batch_cache = {}
    num_test_batches = min(3, metadata["num_batches"])

    for batch_idx in range(num_test_batches):
        batch_file = real_cache_path / f"batch_{batch_idx:06d}.pt"
        if batch_file.exists():
            batch_data = torch.load(batch_file, map_location="cpu")
            batch_cache[batch_idx] = batch_data
            print(f"    - Loaded batch {batch_idx}: {len(batch_data)} images")

            # Check a sample
            if len(batch_data) > 0:
                sample_key = list(batch_data.keys())[0]
                image_id, image_path, image_emb, caption_data_list = batch_data[sample_key]
                print(f"      Sample: image_id={image_id}, path={image_path}, captions={len(caption_data_list)}")
                # Verify image_id is in index
                assert image_id in image_index, f"Image ID {image_id} should be in index"

    print("✓ Real cache test passed")


def test_collate_function():
    """Test collate_coco_batch function."""
    print("Testing collate function...")

    # Create mock batch items with cached embeddings (always present in new implementation)
    batch_items = []
    for i in range(3):
        item = {
            "image": torch.randn(3, 224, 224),
            "student_text_ids": torch.randint(0, 1000, (64,)),
            "caption": f"Test caption {i}",
            "teacher_image_embeds": torch.randn(768),
            "teacher_text_embeds": torch.randn(768),
            "image_id": i + 1,
            "image_path": f"train/{i + 1:012d}.jpg",
        }
        batch_items.append(item)

    # Test collate function
    result = collate_coco_batch(batch_items)

    # Verify all required fields are present
    assert "student_images" in result, "Should have student_images"
    assert "student_text_ids" in result, "Should have student_text_ids"
    assert "captions" in result, "Should have captions"
    assert "teacher_image_embeds" in result, "Should have teacher_image_embeds"
    assert "teacher_text_embeds" in result, "Should have teacher_text_embeds"
    assert "image_ids" in result, "Should have image_ids"
    assert "image_paths" in result, "Should have image_paths"

    # Verify tensor shapes
    assert result["student_images"].shape == (3, 3, 224, 224), "Student images should be batched"
    assert result["student_text_ids"].shape == (3, 64), "Student text IDs should be batched"
    assert result["teacher_image_embeds"].shape == (3, 768), "Teacher image embeds should be batched"
    assert result["teacher_text_embeds"].shape == (3, 768), "Teacher text embeds should be batched"

    # Verify list lengths
    assert len(result["captions"]) == 3, "Should have 3 captions"
    assert len(result["image_ids"]) == 3, "Should have 3 image IDs"
    assert len(result["image_paths"]) == 3, "Should have 3 image paths"

    # Verify content
    assert result["image_ids"] == [1, 2, 3], "Image IDs should match"
    assert all("train/" in path for path in result["image_paths"]), "All paths should contain 'train/'"

    print("✓ Collate function test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing COCOCaptionDataset with new batch-based cache format")
    print("=" * 70)
    print()

    try:
        test_cache_loading()
        print()
        test_batch_loading()
        print()
        test_embedding_retrieval()
        print()
        test_collate_function()
        print()
        test_cache_with_real_data()
        print()
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
