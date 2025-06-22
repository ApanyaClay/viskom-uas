import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time


class MultiPageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Multi-Halaman Pengolahan Citra")
        self.geometry("1000x900") # Ukuran jendela

        # --- Membuat Navbar Frame ---
        navbar_frame = tk.Frame(self, bg="#333333", height=50)
        navbar_frame.pack(side="top", fill="x")

        btn_home_nav = tk.Button(navbar_frame, text="Home",
                                 command=lambda: self.show_frame("HomePage"),
                                 bg="#555555", fg="white", font=("Arial", 12), bd=0, padx=15, pady=10)
        btn_home_nav.pack(side="left", padx=10, pady=5)

        btn_uas_nav = tk.Button(navbar_frame, text="UAS",
                                command=lambda: self.show_frame("UASPage"),
                                bg="#555555", fg="white", font=("Arial", 12), bd=0, padx=15, pady=10)
        btn_uas_nav.pack(side="left", padx=10, pady=5)

        btn_about_nav = tk.Button(navbar_frame, text="About",
                                  command=lambda: self.show_frame("AboutPage"),
                                  bg="#555555", fg="white", font=("Arial", 12), bd=0, padx=15, pady=10)
        btn_about_nav.pack(side="left", padx=10, pady=5)

        # --- Membuat Container untuk Semua Halaman ---
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomePage, UASPage, AboutPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

# --- Halaman Home ---
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.current_image1_path = None
        self.current_image2_path = None
        self.processed_images_tk = []
        # --- Membuat Canvas dan Scrollbar ---
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas) # Ini akan menampung semua konten

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bagian UI lainnya tetap sama
        tk.Label(self.scrollable_frame, text="Selamat Datang di Halaman Utama!", font=("Arial", 24, "bold"),
                 bg="#f0f0f0").pack(pady=20)

        input_frame1 = ttk.Frame(self.scrollable_frame)
        input_frame1.pack(pady=10)
        tk.Label(input_frame1, text="Gambar 1:", font=("Arial", 16), bg="#f0f0f0").pack(side="left", padx=10)
        self.choose_image1_button = tk.Button(input_frame1, text="Pilih Gambar 1",
                                             command=lambda: self.select_image(1),
                                             font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.choose_image1_button.pack(side="left", padx=5)

        self.image1_label = tk.Label(self.scrollable_frame, bg="#f0f0f0")
        self.image1_label.pack(pady=5)

        input_frame2 = ttk.Frame(self.scrollable_frame)
        input_frame2.pack(pady=10)

        tk.Label(input_frame2, text="Gambar 2:", font=("Arial", 16), bg="#f0f0f0").pack(side="left", padx=10)
        self.choose_image2_button = tk.Button(input_frame2, text="Pilih Gambar 2",
                                             command=lambda: self.select_image(2),
                                             font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.choose_image2_button.pack(side="left", padx=5)

        self.image2_label = tk.Label(self.scrollable_frame, bg="#f0f0f0")
        self.image2_label.pack(pady=5)

        self.process_button = tk.Button(self.scrollable_frame, text="Proses Gambar",
                                        command=self.process_image_pipeline,
                                        font=("Arial", 14), bg="#008CBA", fg="white", padx=15, pady=10, state=tk.DISABLED)
        self.process_button.pack(pady=20)

        self.results_frame = ttk.Frame(self.scrollable_frame, style="Card.TFrame")
        self.results_frame.pack(pady=20, padx=20, fill="both", expand=True) # Pastikan expand dan fill
        
        s = ttk.Style()
        s.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1, bordercolor="#cccccc")

        
        # Binding mouse wheel untuk scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel) # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel) # Linux scroll down

    def _on_mousewheel(self, event):
        # Untuk Windows/macOS
        if event.delta:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # Untuk Linux
        elif event.num == 4: # Scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5: # Scroll down
            self.canvas.yview_scroll(1, "units")

    def _on_frame_configure(self, event):
        # Perbarui scrollregion untuk mencakup seluruh bbox dari scrollable_frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Juga, sesuaikan lebar jendela di dalam canvas agar sama dengan lebar canvas
        # Ini mencegah scrollbar horizontal muncul ketika tidak diperlukan karena lebar konten tidak diatur
        canvas_width = event.width # Lebar scrollable_frame
        self.canvas.itemconfig(self.canvas_window_id, width=canvas_width)

    # Metode select_image tetap sama
    def select_image(self, image_num):
        file_path = filedialog.askopenfilename(
            title=f"Pilih Gambar {image_num}",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*"))
        )
        if file_path:
            if image_num == 1:
                self.current_image1_path = file_path
                self.display_image(file_path, label_widget=self.image1_label)
            elif image_num == 2:
                self.current_image2_path = file_path
                self.display_image(file_path, label_widget=self.image2_label)
            
            self.check_process_button_state()
            self.clear_results()

    # Metode display_image (sedikit disederhanakan karena _on_frame_configure yang akan memperbarui scrollregion)
    def display_image(self, image_input, title="Gambar", is_original=False, label_widget=None):
        try:
            image_pil = None

            if isinstance(image_input, str):
                original_image_pil = Image.open(image_input)
                if original_image_pil.mode != 'RGB':
                    original_image_pil = original_image_pil.convert('RGB')
                image_pil = original_image_pil

            elif isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
                elif len(image_input.shape) == 2:
                    image_pil = Image.fromarray(image_input)
                    if image_pil.mode != 'RGB':
                         image_pil = image_pil.convert('RGB')
                else:
                    messagebox.showerror("Error", "Tipe array gambar tidak didukung untuk tampilan.")
                    return
            else:
                messagebox.showerror("Error", "Tipe input gambar tidak didukung.")
                return
            
            if image_pil is None:
                messagebox.showerror("Error", "Gagal mengonversi gambar ke format tampilan.")
                return

            max_width = 250
            max_height = 250

            if not is_original:
                max_width = 600
                max_height = 600

            original_width, original_height = image_pil.size

            if original_width > max_width or original_height > max_height:
                ratio = min(max_width / original_width, max_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                resized_image = image_pil.resize((new_width, new_height), Image.LANCZOS)
            else:
                resized_image = image_pil

            tk_image = ImageTk.PhotoImage(resized_image)
            
            if label_widget:
                label_widget.config(image=tk_image)
                label_widget.image = tk_image
            else:
                result_card = ttk.Frame(self.results_frame, style="Card.TFrame", padding=(10, 10, 10, 10))
                result_card.pack(pady=10, fill="x")

                ttk.Label(result_card, text=title, font=("Arial", 16, "bold"), background="#ffffff", wraplength=600).pack(pady=5)
                
                result_image_label = ttk.Label(result_card, background="#ffffff")
                result_image_label.pack(pady=5)
                result_image_label.config(image=tk_image)
                result_image_label.image = tk_image
                self.processed_images_tk.append(tk_image)

            # Baris update_idletasks dan config scrollregion dihilangkan dari sini
            # karena _on_frame_configure akan menanganinya secara otomatis
            # saat ukuran scrollable_frame berubah.

        except Exception as e:
            tk.messagebox.showerror("Error", f"Gagal memuat atau menampilkan gambar: {e}. Pastikan file gambar valid.")
            if label_widget:
                label_widget.config(image="")


    # Metode check_process_button_state dan clear_results tetap sama
    def check_process_button_state(self):
        if self.current_image1_path and self.current_image2_path:
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)

    def clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.processed_images_tk = []

    # Metode _on_canvas_configure untuk mengatur visibilitas scrollbar
    def _on_canvas_configure(self, event):
        # Ukuran canvas saat ini
        canvas_width = event.width
        canvas_height = event.height
        
        # Atur lebar jendela internal canvas agar sama dengan lebar canvas
        self.canvas.itemconfig(self.canvas_window_id, width=canvas_width)

        # Dapatkan bbox semua item di canvas
        bbox = self.canvas.bbox("all")
        if bbox: # Pastikan bbox tidak kosong
            # Perbarui scrollregion berdasarkan bbox semua item
            self.canvas.configure(scrollregion=bbox)

            # Sembunyikan/tampilkan scrollbar Y
            if bbox[3] > canvas_height: # Jika tinggi konten > tinggi canvas
                self.scroll_y.grid(row=0, column=1, sticky="ns")
            else:
                self.scroll_y.grid_forget()
            
            # Sembunyikan/tampilkan scrollbar X
            if bbox[2] > canvas_width: # Jika lebar konten > lebar canvas
                self.scroll_x.grid(row=1, column=0, sticky="ew")
            else:
                self.scroll_x.grid_forget()
        else: # Jika tidak ada konten, sembunyikan semua scrollbar
            self.scroll_y.grid_forget()
            self.scroll_x.grid_forget()

    # --- Fungsi-fungsi Pemrosesan Citra (tetap sama seperti balasan sebelumnya) ---
    def peningkatan_kontras(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 3:
            gray_image22 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image22 = image
        if gray_image22.dtype != np.uint8:
            gray_image22 = np.uint8(gray_image22)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image22)

    def thresholding(self, image, threshold=127):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.dtype != np.uint8:
            image = np.uint8(image)
        _, binary_threshold = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary_threshold

    def edge_detection(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(image, threshold1=100, threshold2=200)

    def sift(self, image, image2):
        sift = cv2.SIFT_create()
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)
        sift_result = cv2.drawKeypoints(
            image2, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return sift_result, keypoints

    def gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    def sobel(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(np.float32(sobel_x), np.float32(sobel_y))
        return sobel_combined

    def prewitt(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_prewitt_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_prewitt_y)
        prewitt_combined = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))
        return prewitt_combined

    def closing(self, image):
        if image.dtype != np.uint8:
            image = np.uint8(image)
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closing

    def opening(self, image):
        if image.dtype != np.uint8:
            image = np.uint8(image)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opening

    def skeletonisasi(self, image):
        skel = np.zeros(image.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_image = image.copy()

        while True:
            open_img = cv2.morphologyEx(temp_image, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(temp_image, open_img)
            eroded = cv2.erode(temp_image, element)
            skel = cv2.bitwise_or(skel, temp)
            temp_image = eroded.copy()
            
            if cv2.countNonZero(temp_image) == 0:
                break
        return skel

    def orb(self, image, image2):
        orb = cv2.ORB_create()
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        keypoints1, descriptors1 = orb.detectAndCompute(image_gray, None)
        orb_result = cv2.drawKeypoints(
            image2, keypoints1, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return orb_result, keypoints1

    def dfo(self, image1, image2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)
        
        if des1 is None or des2 is None:
            messagebox.showinfo("Informasi", "Tidak cukup fitur untuk pencocokan ORB.")
            h, w = image1.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        num_matches_to_draw = min(50, len(good_matches))
        # Draw matches pada gambar berwarna asli jika memungkinkan
        # Pastikan image1 dan image2 di sini adalah versi berwarna jika ingin gambar berwarna
        matched_image = cv2.drawMatches(
            image1, kp1, # Convert grayscale to BGR for drawing if original was BGR
            image2, kp2, # Same for image2
            good_matches[:num_matches_to_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        return matched_image, kp1, kp2, good_matches

    def process_image_pipeline(self):
        if not self.current_image1_path or not self.current_image2_path:
            messagebox.showerror("Error", "Pilih kedua gambar terlebih dahulu!")
            return

        self.clear_results()

        try:
            # Baca gambar pertama dan kedua
            inputgambar1_cv = cv2.imread(self.current_image1_path)
            inputgambar2_cv = cv2.imread(self.current_image2_path)

            if inputgambar1_cv is None:
                messagebox.showerror("Error", "Gagal membaca Gambar 1. Pastikan file gambar valid.")
                return
            if inputgambar2_cv is None:
                messagebox.showerror("Error", "Gagal membaca Gambar 2. Pastikan file gambar valid.")
                return
            
            # Tampilkan gambar asli pertama dan kedua sebagai header
            self.display_image(inputgambar1_cv, title="Gambar Asli 1", is_original=True, label_widget=self.image1_label)
            self.display_image(inputgambar2_cv, title="Gambar Asli 2", is_original=True, label_widget=self.image2_label)


            results = []
            
            # --- Alur Pemrosesan (Menggunakan inputgambar1_cv untuk sebagian besar operasi) ---
            # Input untuk peningkatan_kontras (image_gray)
            langkah1_1 = self.peningkatan_kontras(inputgambar1_cv)
            results.append((langkah1_1, "Peningkatan Kontras (Gambar 1)"))

            langkah1_2_gray = langkah1_1
            results.append((langkah1_2_gray, "Grayscale (Gambar 1)"))

            langkah2_1 = self.edge_detection(langkah1_2_gray)
            results.append((langkah2_1, "Edge Detection (Gambar 1 - Canny)"))

            langkah2_2 = self.sift(langkah1_2_gray, inputgambar1_cv) # SIFT pada Gambar 1
            results.append((langkah2_2[0], f"SIFT Result (Gambar 1 - {len(langkah2_2[1])} Keypoints)"))

            langkah3_1 = self.gaussian_blur(langkah1_2_gray) # Gaussian Blur pada Gambar 1
            results.append((langkah3_1, "Gaussian Blur (Gambar 1)"))

            langkah3_2 = self.sobel(langkah3_1) # Sobel pada Gambar 1
            results.append((langkah3_2, "Sobel (Gambar 1)"))

            langkah3_3 = self.prewitt(langkah3_1) # Prewitt pada Gambar 1
            results.append((langkah3_3, "Prewitt (Gambar 1)"))
            
            thresholded_img_for_morph = self.thresholding(langkah1_2_gray)
            langkah4_1 = self.opening(thresholded_img_for_morph)
            results.append((langkah4_1, "Opening (Gambar 1 - setelah Thresholding)"))

            langkah4_2 = self.closing(langkah4_1)
            results.append((langkah4_2, "Closing (Gambar 1 - setelah Opening)"))

            langkah4_3 = self.skeletonisasi(langkah4_2)
            results.append((langkah4_3, "Skeletonisasi (Gambar 1)"))

            langkah5_1 = self.orb(langkah1_2_gray, inputgambar1_cv) # ORB pada Gambar 1
            results.append((langkah5_1[0], f"ORB Result (Gambar 1 - {len(langkah5_1[1])} Keypoints)"))

            # --- Pencocokan Fitur ORB (DFO) - Menggunakan Gambar 1 dan Gambar 2 ---
            langkah5_2 = self.dfo(inputgambar1_cv, inputgambar2_cv) # DFO menggunakan kedua gambar
            results.append((langkah5_2[0], f"Pencocokan Fitur ORB (Gambar 1 - {len(langkah5_2[1])} Keypoints & Gambar 2 - {len(langkah5_2[2])} Keypoints) Jumlah Pencocokan: {len(langkah5_2[3])}"))


            # Tampilkan semua hasil secara berurutan
            for img_result, title_result in results:
                self.display_image(img_result, title=title_result)

        except Exception as e:
            messagebox.showerror("Error Pemrosesan", f"Terjadi kesalahan saat memproses gambar: {e}. Periksa konsistensi tipe data gambar.")

# --- Halaman UAS ---
class UASPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="lightgreen")
        self.controller = controller
        
        self.original_image_cv = None 
        self.current_display_image = None

        # --- Membuat Canvas dan Scrollbar untuk halaman UAS juga ---
        self.canvas = tk.Canvas(self, bg="lightgreen")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="lightgreen") # Ini akan menampung semua konten

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # --- Konten Halaman UAS (dipindahkan ke self.scrollable_frame) ---
        label = tk.Label(self.scrollable_frame, text="Ini adalah Halaman UAS", font=("Helvetica", 24), bg="lightgreen")
        label.pack(pady=30, padx=50)

        uas_info_label = tk.Label(self.scrollable_frame, text="Informasi terkait Ujian Akhir Semester akan ditampilkan di sini.", 
                                  font=("Helvetica", 14), bg="lightgreen")
        uas_info_label.pack(pady=20)

        btn_select_image = tk.Button(self.scrollable_frame, text="Pilih Gambar untuk UAS", command=self.select_image_uas, font=("Helvetica", 12))
        btn_select_image.pack(pady=10)

        self.image_label = tk.Label(self.scrollable_frame, bg="lightgreen")
        self.image_label.pack(pady=20)

        # Binding mouse wheel untuk scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def select_image_uas(self):
        file_path = filedialog.askopenfilename(
            title="Pilih File Gambar untuk UAS",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"), ("All files", "*.*")]
        )
        if file_path:
            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None:
                    raise ValueError("Gagal memuat gambar. Pastikan file bukan korup.")
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                self.original_image_cv = img_cv.copy() 
                self._display_image_on_gui(self.original_image_cv, "Gambar Asli Dimuat di UAS")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat gambar: {e}")
                self.image_label.config(image='')
                self.current_display_image = None
                self.original_image_cv = None

    def _display_image_on_gui(self, cv_image, status_text="Menampilkan Hasil"):
        if cv_image is None:
            self.image_label.config(image='')
            self.current_display_image = None
            return

        if cv_image.dtype != np.uint8:
            cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        if len(cv_image.shape) == 2:
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv_image
            
        max_size = (400, 400)
        img_pil = Image.fromarray(display_image)
        img_pil.thumbnail(max_size, Image.LANCZOS)

        self.current_display_image = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.current_display_image)
        self.image_label.image = self.current_display_image
        # Tidak ada status_label di halaman UAS, jadi baris ini bisa dihapus atau di-comment
        # self.status_label.config(text=f"Status: {status_text}") 
        self.update_idletasks()

class AboutPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="lightcoral")
        self.controller = controller
        
        label = tk.Label(self, text="Tentang Aplikasi Ini", font=("Helvetica", 24), bg="lightcoral")
        label.pack(pady=50, padx=50)

        about_text = ("Aplikasi ini adalah contoh sederhana GUI Python multi-halaman.\n"
                      "Dibuat dengan Tkinter untuk tujuan demonstrasi.")
        about_info_label = tk.Label(self, text=about_text, font=("Helvetica", 14), bg="lightcoral")
        about_info_label.pack(pady=20)

if __name__ == "__main__":
    app = MultiPageApp()
    app.mainloop()