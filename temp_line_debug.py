def detect_line(self, binary):

    debug = self.pub_debug.get_num_connections() > 0

    if debug:
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    ys = np.linspace(
        int(self.height * 0.6),
        int(self.height * 0.9),
        6
    ).astype(int)

    pts = []

    for y in ys:

        row = binary[y]

        xs = np.where(row > 0)[0]

        if len(xs) < 10:
            continue

        x = int(np.mean(xs))

        pts.append((x, y))

        if debug:
            cv2.circle(dbg, (x, y), 4, (0,255,255), -1)

    if len(pts) < 3:
        if debug:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
        return None

    pts = np.array(pts, np.float32)

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

    y1 = int(self.height * 0.78)
    y2 = int(self.height * 0.55)

    x1 = int((y1 - y0) * vx / vy + x0)
    x2 = int((y2 - y0) * vx / vy + x0)

    if debug:

        cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 3)

        for y in ys:
            cv2.line(dbg, (0,y), (self.width,y), (255,0,0), 1)

        angle = np.degrees(np.arctan2(vy, vx))

        cv2.putText(
            dbg,
            f"angle={angle:.1f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

    return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)
