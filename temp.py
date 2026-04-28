    def process(self, image, frame_id=None):
        t0 = time.perf_counter()

        # ---------- ROI ----------
        roi = image[self.y1:self.y2, self.x1:self.x2]

        t1 = time.perf_counter()

        # ---------- RGB -> HSV ----------
        roi = image[self.y1:self.y2, self.x1:self.x2]

        r = roi[:,:,0]
        g = roi[:,:,1]
        b = roi[:,:,2]


        t2 = time.perf_counter()

        # ---------- MASK ----------
        if self.color == ColorLine.yellow:
            mask = (
                (r > 140) &
                (g > 120) &
                (b < 130) &
                (r > g)
            ).astype(np.uint8) * 255
        else:
            mask = (
                (r > 170) &
                (g > 170) &
                (b > 170) &
                (np.abs(r-g) < 35) &
                (np.abs(r-b) < 35) &
                (np.abs(g-b) < 35)
            ).astype(np.uint8) * 255

        t3 = time.perf_counter()

        # ---------- LINE DETECTION ----------
        result = self.detect_line_fast(mask)

        t4 = time.perf_counter()

        # ---------- TIMING ----------
        total = (t4 - t0) * 1000
        roi_t = (t1 - t0) * 1000
        hsv_t = (t2 - t1) * 1000
        mask_t = (t3 - t2) * 1000
        detect_t = (t4 - t3) * 1000

        print(
            f"ROI:{roi_t:.1f} ms | "
            f"HSV:{hsv_t:.1f} ms | "
            f"MASK:{mask_t:.1f} ms | "
            f"LINE:{detect_t:.1f} ms | "
            f"TOTAL:{total:.1f} ms"
        )

        if result is None:
            return {"valid": False, "line": None, "k": None, "b": None}

        line, k, b = result
        return {"valid": True, "line": line, "k": k, "b": b}


    def detect_line_fast(self, binary):
        t0 = time.perf_counter()

        h, w = binary.shape

        # scanlines заранее лучше хранить в self.scan_ys
        ys = self.scan_ys

        pts = []

        for y in ys:
            row = binary[y]

            xs = cv2.findNonZero(row.reshape(1, -1))

            if xs is None or len(xs) < 5:
                continue

            x = int(np.mean(xs[:, 0, 0]))
            pts.append((x, y))

        t1 = time.perf_counter()

        if len(pts) < 2:
            print(f"SCAN:{(t1-t0)*1000:.1f} ms | NO LINE")
            return None

        pts = np.array(pts, dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            pts,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        if abs(vy) < 1e-5:
            return None

        y1 = int(h * 0.85)
        y2 = int(h * 0.45)

        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        k = float(vx / vy)
        b = float(x0 - k * y0)

        t2 = time.perf_counter()

        print(
            f"SCAN:{(t1-t0)*1000:.1f} ms | "
            f"FIT:{(t2-t1)*1000:.1f} ms"
        )

        return (x1 + self.x1, y1 + self.y1,
                x2 + self.x1, y2 + self.y1), k, b

        roi_h = self.y2 - self.y1

        self.scan_ys = np.linspace(
            int(roi_h * 0.55),
            int(roi_h * 0.95),
            9
        ).astype(int)
