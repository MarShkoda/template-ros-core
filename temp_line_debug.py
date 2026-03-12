def detect_line(self, binary):
        edges = cv2.Canny(binary, 80, 160)

        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=20,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is None:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        pts = []

        for l in lines:
            x1, y1, x2, y2 = l[0]

            # все линии Hough (синие)
            cv2.line(dbg, (x1, y1), (x2, y2), (255,0,0), 1)

            slope = (y2 - y1) / (x2 - x1 + 1e-5)

            if abs(x2 - x1) < 5 or abs(slope) > 0.3:

                # линии прошедшие фильтр (зелёные)
                cv2.line(dbg, (x1, y1), (x2, y2), (0,255,0), 2)

                pts.append((x1, y1))
                pts.append((x2, y2))

        if len(pts) < 6:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        pts = np.array(pts, np.float32)

        # точки fitLine (желтые)
        for p in pts:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 3, (0,255,255), -1)

        vx, vy, x0, y0 = cv2.fitLine(
            pts,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        # защита от деления на 0
        if abs(vy) < 1e-5:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        y1 = int(self.height * 0.78)
        y2 = int(self.height * 0.55)

        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        # итоговая линия (красная)
        cv2.line(dbg, (x1, y1), (x2, y2), (0,0,255), 3)

        # горизонтали ROI
        cv2.line(dbg, (0, y1), (self.width, y1), (255,255,0), 1)
        cv2.line(dbg, (0, y2), (self.width, y2), (255,255,0), 1)

        # угол линии
        angle = np.degrees(np.arctan2(vy, vx))
        cv2.putText(
            dbg,
            f"angle={angle:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        return (x1, y1, x2, y2), float(vx/vy), float(x0 - (vx/vy) * y0)
