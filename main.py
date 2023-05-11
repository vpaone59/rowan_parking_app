from parkingLotDetector import ParkingLotDetector

if __name__ == '__main__':
    detector = ParkingLotDetector('video.mp4', 'CarParkPos')
    detector.run()