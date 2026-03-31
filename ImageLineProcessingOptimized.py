collage_store = {}  # {frame_id: {"orig": img, "yellow": dbg_y, "white": dbg_w}}

class ImageLineProcessingOptimized:
    def __init__(self, width, height, color: 'ColorLine', save_debug_dir="debug_photos"):
        self.width = width
        self.height = height
        self.color = color
        self.bridge = CvBridge()
        self.save_debug_dir = save_debug_dir
        os.makedirs(self.save_debug_dir, exist_ok=True)

        if color == ColorLine.yellow:
            #self.lower = np.array([25,140,100], np.uint8)
            #self.upper = np.array([45,255,255], np.uint8)
            self.lower = np.array([20, 80, 80])
            self.upper = np.array([50, 255, 255])
            self.left, self.right = 0.0, 0.8
            self.pub_debug = rospy.Publisher("/yellow_lane_debug", Image, queue_size=1)
            self.pub_debug_mask = rospy.Publisher("/yellow_mask", Image, queue_size=1)
            self.pub_debug_roi = rospy.Publisher("/yellow_roi", Image, queue_size=1)


        else:
            self.lower = np.array([0,0,150], np.uint8)
            self.upper = np.array([180,100,255], np.uint8)
            self.left, self.right = 0.2, 1.0
            self.pub_debug = rospy.Publisher("/white_lane_debug", Image, queue_size=1)
            self.pub_debug_mask = rospy.Publisher("/white_mask", Image, queue_size=1)
            self.pub_debug_roi = rospy.Publisher("/white_roi", Image, queue_size=1)


        self._roi_mask = None
        self.image = None

    def _get_roi_mask(self, target_shape=None):
        """Create ROI mask, optionally resizing to target shape"""
        if self._roi_mask is None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertices = np.array([[
                (int(self.width * self.left),  int(self.height * 2/3)),
                (int(self.width * self.left),  int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 2/3)),
            ]], np.int32)
            cv2.fillPoly(mask, vertices, 255)
            self._roi_mask = mask
        
        # Resize mask if target shape is provided and different
        if target_shape is not None:
            if target_shape[:2] != self._roi_mask.shape[:2]:
                return cv2.resize(self._roi_mask, (target_shape[1], target_shape[0]))
        
        return self._roi_mask

    def region_selection(self, color_mask):
        roi_mask = self._get_roi_mask(color_mask.shape)
        return cv2.bitwise_and(color_mask, roi_mask)

    def detect_line(self, binary, orig_image=None, frame_id=None):
        debug = True #self.pub_debug.get_num_connections() > 0

        if debug:
            dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        ys = np.linspace(
            int(self.height * 0.6),
            int(self.height * 0.9),
            9
        ).astype(int)

        pts = []

        for y in ys:
            row = binary[y]
            xs = np.where(row > 0)[0]
            if len(xs) < 10:
                continue
            x = int(np.median(xs))
            pts.append((x, y))
            if debug:
                cv2.circle(dbg, (x, y), 4, (0,255,255), -1)

        if len(pts) < 2:
            if debug:
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
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
            k = float(vx/vy)
            angle = np.degrees(math.atan(k))
            text_y = 40
            step_y = 20
            cv2.putText(dbg, f"angle={angle:.1f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"vx={vx:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"vy={vy:.1f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"k=vx/vy={k:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            angle_rad = math.atan(k)
            text_y=text_y+step_y
            cv2.putText(dbg, f"angle_rad=math.atan(k)={angle_rad:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        # ---- Сохраняем для коллажа ----
        if orig_image is not None and frame_id is not None and frame_id < 100:
            # сохраняем в глобальный словарь
            if frame_id not in collage_store:
                collage_store[frame_id] = {}
            if self.color == ColorLine.yellow:
                collage_store[frame_id]["yellow"] = dbg
            else:
                collage_store[frame_id]["white"] = dbg
            collage_store[frame_id]["orig"] = orig_image

            # если все три изображения готовы, сохраняем коллаж
            parts = collage_store[frame_id]
            if all(k in parts for k in ["orig","yellow","white"]):
                # делаем коллаж: ширина = 3*ширина, высота = высота
                collage = np.zeros((self.height, self.width*3, 3), np.uint8)
                collage[:, :self.width] = parts["yellow"]
                collage[:, self.width:2*self.width] = parts["orig"]
                collage[:, 2*self.width:] = parts["white"]
                save_path = os.path.join(self.save_debug_dir, f"{frame_id:06d}.png")
                cv2.imwrite(save_path, collage)
                #rospy.loginfo(f"path {save_path}")
                # удаляем из словаря, чтобы не переполнять память
                del collage_store[frame_id]

        #return (x1, y1), (x2, y2)
        return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)

    def process(self, image, frame_id):
        self.image = cv2.blur(image, (3,3))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(self.image, self.lower, self.upper)
        self.pub_debug_mask.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))
        roi = self.region_selection(mask)
        self.pub_debug_roi.publish(self.bridge.cv2_to_imgmsg(roi, encoding="mono8"))
        _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        result = self.detect_line(roi, self.image, frame_id)
        if result is None:
            return {"valid": False, "line": None, "k": None, "b": None}

        line, k, b = result
        return {"valid": True, "line": line, "k": k, "b": b}

    def x_at_y(self, model, y):
        if not model["valid"]:
            return None
        return model["k"] * y + model["b"]

    def draw(self, image, model, color=(0,255,0), thickness=2):
        if not model["valid"]:
            return image
        x1,y1,x2,y2 = model["line"]
        return cv2.line(image, (x1,y1), (x2,y2), color, thickness)
