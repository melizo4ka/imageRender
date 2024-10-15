import concurrent
import threading
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import time


load_finished_event = threading.Event()


class ImageManager:
    def __init__(self):
        self.images = {}
        self.lock = threading.Lock()
        self.max_display_size = (800, 600)

    def load_image(self, filepath):
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Error loading {filepath}: Image could not be opened.")
            filename = os.path.basename(filepath)

            # Store the uncompressed image (NumPy array)
            with self.lock:
                self.images[filename] = {
                    "compressed": image,  # The loaded image is already in NumPy array format
                    "uncompressed": image
                }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def load_images_sequential(self, folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.endswith((".jpg", ".jpeg"))]
        for filepath in image_files:
            self.load_image(filepath)

    def load_images_background(self, folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.endswith((".jpg", ".jpeg"))]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.load_image, image_files)
        load_finished_event.set()

    def resize_image(self, image):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        if width > height:
            new_width = self.max_display_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.max_display_size[1]
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(image, (new_width, new_height))

    async def render_image(self, filename):
        await asyncio.sleep(0)  # This allows the event loop to continue
        if filename in self.images:
            # print(f"Rendering {filename}...")
            resized_image = self.resize_image(self.images[filename]["compressed"])
            cv2.imshow(filename, resized_image)
            cv2.waitKey(2000)  # Display for 2000ms (2 seconds)
            cv2.destroyAllWindows()
        else:
            print(f"Image {filename} not found.")

    async def render_images_list(self, file_list):
        tasks = [self.render_image(filename) for filename in file_list]
        await asyncio.gather(*tasks)

    def get_compressed_image(self, filename):
        if filename in self.images:
            return self.images[filename]["compressed"]
        else:
            raise ValueError("Image not found.")

    def display_image(self, filename):
        image = self.get_compressed_image(filename)
        resized_image = self.resize_image(image)
        cv2.imshow(filename, resized_image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


def background_image_loader(image_manager, folder_path):
    thread = threading.Thread(target=image_manager.load_images_background, args=(folder_path,))
    thread.daemon = True  # Daemon thread will exit when main program exits
    thread.start()


if __name__ == "__main__":
    img_manager = ImageManager()
    folder_path = "images/"

    example_image_1 = "cat.jpg"
    example_image_2 = "leaf.jpg"
    example_image_3 = "cloud.jpg"

    # Sequential image loader
    print("Starting sequential image loading...")

    start_time_seq = time.time()
    img_manager.load_images_sequential(folder_path)

    start_render_seq = time.time()
    img_manager.display_image(example_image_1)
    img_manager.display_image(example_image_2)
    img_manager.display_image(example_image_3)
    end_render_seq = time.time()
    print(f"Sequential rendering time: {end_render_seq - start_render_seq:.4f} seconds")

    end_time_seq = time.time()
    time_sequential = end_time_seq - start_time_seq

    # Reset the images dictionary before running the parallel load
    img_manager.images = {}

    # Parallel image loader - multithreading for loading and asyncio for rendering
    print("Starting parallel image loading...")

    # print("Starting background image loading...")
    start_time_par = time.time()
    background_image_loader(img_manager, folder_path)
    load_finished_event.wait()

    async def main():
        start_render_par = time.time()
        # print("Rendering a list of images asynchronously...")
        await img_manager.render_images_list([example_image_1, example_image_2, example_image_3])
        end_render_par = time.time()
        print(f"Parallel rendering time: {end_render_par - start_render_par:.4f} seconds")

    asyncio.run(main())
    end_time_par = time.time()
    time_parallel = end_time_par - start_time_par

    # Comparison
    print(f"Sequential time: {time_sequential:.4f} seconds")
    print(f"Parallel time: {time_parallel:.4f} seconds")

