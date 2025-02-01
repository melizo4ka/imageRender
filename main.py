import cv2
import time
import os
import asyncio
import multiprocessing
from multiprocessing import Manager


def load_image_worker(filepath, shared_dict, lock):
    try:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Error loading {filepath}: Image could not be opened.")
        filename = os.path.basename(filepath)
        with lock:
            shared_dict[filename] = image
    except Exception as e:
        print(f"Error loading {filepath}: {e}")


class ImageManager:
    def __init__(self):
        self.manager = Manager()
        self.images = self.manager.dict()
        self.lock = Manager().Lock()
        self.max_display_size = (800, 600)
        self.simulate_only = True

    def load_image(self, filepath):
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Error loading {filepath}: Image could not be opened.")
            filename = os.path.basename(filepath)
            self.images[filename] = image
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def load_images_sequential(self, folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.endswith((".jpg", ".jpeg"))]
        for filepath in image_files:
            self.load_image(filepath)

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

    def get_compressed_image(self, filename):
        if filename in self.images:
            return self.images[filename]
        else:
            raise ValueError("Image not found.")

    def display_image(self, filename):
        self.flip_image(filename)
        image = self.get_compressed_image(filename)
        resized_image = self.resize_image(image)
        if not self.simulate_only:
            cv2.imshow(filename, resized_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            # simulating display time
            time.sleep(1)

    def load_images_parallel(self, folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.endswith((".jpg", ".jpeg"))]

        processes = []
        for filepath in image_files:
            p = multiprocessing.Process(
                target=load_image_worker,
                args=(filepath, self.images, self.lock)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def flip_image(self, filename, flip_code=-1):
        if filename in self.images:
            self.images[filename] = cv2.flip(self.images[filename], flip_code)
        else:
            print(f"Image {filename} not found.")

    async def render_image(self, filename):
        try:
            if filename in self.images:
                self.flip_image(filename)
                resized_image = self.resize_image(self.images[filename])
                if not self.simulate_only:
                    cv2.imshow(filename, resized_image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                else:
                    # simulating display time
                    await asyncio.sleep(1)
            else:
                print(f"Image {filename} not found.")
        except Exception as e:
            print(f"Error rendering {filename}: {e}")

    async def render_images_list(self, file_list):
        tasks = [self.render_image(filename) for filename in file_list]
        await asyncio.gather(*tasks)


def background_image_loader(image_manager, folder_path):
    image_manager.load_images_parallel(folder_path)


if __name__ == "__main__":
    print("Starting the execution...")
    to_print = False
    NUM_RUNS = 10

    speedup_render = []
    speedup_total = []

    folder_path = "images/"
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg'))]
    image_list = []
    for i, image_file in enumerate(image_files, start=1):
        image_list.append(image_file)

    for run in range(NUM_RUNS):
        img_manager = ImageManager()
        img_manager.simulate_only = True

        # SEQUENTIAL IMAGE LOADER
        start_time_seq = time.time()
        img_manager.load_images_sequential(folder_path)
        start_render_seq = time.time()
        for image in image_list:
            img_manager.display_image(image)
        end_render_seq = time.time()
        end_time_seq = time.time()
        time_sequential = end_time_seq - start_time_seq
        if to_print:
            print(f"Sequential rendering time: {end_render_seq - start_render_seq:.4f} seconds")

        # resetting the images dictionary
        img_manager.images.clear()

        # PARALLEL IMAGE LOADER
        start_time_par = time.time()
        background_image_loader(img_manager, folder_path)

        async def render_parallel_images():
            start_render_par = time.time()
            await img_manager.render_images_list(image_list)
            end_render_par = time.time()
            if to_print:
                print(f"Parallel rendering time: {end_render_par - start_render_par:.4f} seconds")
            su_render = (end_render_seq - start_render_seq) / (end_render_par - start_render_par)
            speedup_render.append(su_render)

        asyncio.run(render_parallel_images())
        end_time_par = time.time()
        time_parallel = end_time_par - start_time_par
        img_manager.images.clear()

        # comparison
        if to_print:
            print(f"Sequential time: {time_sequential:.4f} seconds")
            print(f"Parallel time: {time_parallel:.4f} seconds")

        speedup_total.append(time_sequential / time_parallel)
    print(f"The render speedup is: {speedup_render}")
    print(f"The total speedup is: {speedup_total}")

