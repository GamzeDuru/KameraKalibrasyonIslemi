
import numpy as np
import cv2

# Satranç tahtasının içerdiği iç köşe sayısı (örneğin 8x8 tahta için 7x7)
internal_corners = (7, 7)  # iç köşe sayısı

# Kare boyutu (mm cinsinden ölçülü)
square_size = 50

# Algılanacak desenlerin listesi
object_points = [] #3d dünya görüntüsü için
image_points = [] #2d dünya görüntüsü için

# Dünya koordinatları (x, y, z)
#İç köşelerin 3D dünya koordinatlarını tutmak için -> world_points
#satranç tahtasının iç köşelerinin toplam sayısını (internal_corners[0] * internal_corners[1]) ve her bir iç köşenin x, y ve z koordinatlarını tutacak sütun sayısını (3) belirtir.
# internal_corners[0] * internal_corners[1] -> piksel sayısını verir
world_points = np.zeros((internal_corners[0] * internal_corners[1], 3), np.float32)
#np.mgrid:koordinat matrisi oluşturuluyor.
#0:internal_corners[0], 0:internal_corners[1]] ->  satranç tahtasının iç köşelerinin x ve y koordinatlarını oluşturur.
#T.reshape(-1, 2) -> x ve y koordinatlarının yer değiştirmesini sağlar
#world_points[:, :2] -> satranç tahtasındaki iç köşelerin 2D dünya koordinatlarını oluşturur.Tüm satırları ve 2 sütunu seçer.
# 0: başlangıç indeksi  ,  internel_corners[0] en son nereye kadar gidecek  bizim  internel_corners[0] değerimiz 7 bu yüzden 0-dan 7ye kadar gidecek
#0:internal_corners[0]  -> x koordinatı  0:internal_corners[1]] -> y koordinatı
#T.reshape(-1,2) -> ekrana okutacağımız görselin transpozunu alır ve satır sayısını hesaplar(-1) sütun sayısını belirtir(2)
world_points[:, :2] = np.mgrid[0:internal_corners[0], 0:internal_corners[1]].T.reshape(-1, 2) * square_size


#cv2 sınıfından VideoCapture adında bir nesne oluşturur. 0 değeri -> kaç tane kamera var eğer bir tane ise genelde 0 yazılır.
cap = cv2.VideoCapture(0)

while True:

    #video kaynağından bir kareyi okur.
    #read() fonk. geriye iki değer döndürür.
    #ret : kareyi başarıyla okuyup okuyamadığımızı belirten bir bool değerdir. Kameradan görüntü okunup okunmadığını kontrol etmek için kullanılır.
    #frame: okunan karenin kendisidir.NumPy dizisi olarak temsil edilen görüntü verisidir.
    ret, frame = cap.read()

    #görüntüyü gri renge çevirme işlemi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Satranç tahtasındaki iç köşeleri bul
    # findChessboardCorners() bu fonksiyonumuzda 2 değer döndürmektedir. ret ve corners
    # eğer köşeler bulunduysa ret true dönecektir
    # cornes da köşeleri temsil eder.
    ret, corners = cv2.findChessboardCorners(gray, internal_corners, None)
    print(corners)
    #eğer ret==True ise
    if ret:

        cv2.imwrite("k_not.png" , frame)
        # cornerSubpix() -> Köşeleri daha hassas bir şekilde bul
        # Kendi içerisinde yinelemeli olarak çalışan algoritmalar için durdurma/karar verme ölçütü
        # aralarındaki + işareti -> max iterasyona ulaşabilen(30) veya hata oranı belli bir oranın (0.001) altına düşene kadar devam edecek bir kriter oluşturur.
        # 30 : max iterasyon sayısı, 0.001 :epsilon degeri (iterasyonların devam edebileceği hata toleransını belirler).Hata toleransı 0.001'den küçük olursa iterasyon sona erer.
        '''
        11x11 boyutundaki pencere, özellikle daha küçük veya düşük çözünürlüklü görüntülerde köşelerin konumunu daha doğru bir şekilde belirlemeye yardımcı olabilir.
         Bu, köşelerin bulunması ve konumlarının daha hassas bir şekilde belirlenmesi için yapılan bir iyileştirme adımıdır.
         
         (-1,-1) ise konum iyileştirme işleminin sonlandırılması işlemi
        '''
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_points.append(corners)  #piksel düzeyinde iç köşeleri saklar
        object_points.append(world_points) #dünya koord. iç köşeleri alır

        # Köşeleri ekranda çiz
        #Bu çizme işlemi, cv2.findChessboardCorners() işlevi ile bulunan köşelerin doğruluğunu kontrol etmek  veya görselleştirmek için kullanıır
        #frame:köşelerin çizileceği görüntü
        #corners: Bulunan köşelerin piksel koordinatlarını içeren bir dizi veya yapı.
        #internal_corners: Satranç tahtasının iç köşe sayısı.
        #ret: cv2.findChessboardCorners işlevinden dönen başarı durumunu ifade eden bir değer.

        cv2.drawChessboardCorners(frame, internal_corners, corners, ret)


        # Kamera kalibrasyonunu yap
        #object_points: Dünya koordinatlarında iç köşelerin konumlarını içeren bir liste veya dizi.
        #image_points: Piksel düzeyinde iç köşelerin konumlarını içeren bir liste veya dizi.
        #None :Kamera matrisini ve distorsiyon katsayılarını hesaplamak için kullanılan döngüsel kalibrasyon grid boyutu. Bu durumda bu değer None olarak belirtilmiş, yani varsayılan bir değer kullanılmış.

        '''
        #gray.shape[::-1] -> Şekli alır ve onu tersine çevirir. Neden bu işlemi yapıyoruz ? 
        Çünkü kameradan görüntü alırken yükseklik ve genişlik değerleri yanlışlıkla ters saklanmış
        olabilir. Böyle durumları düzeltmek için kullanırız.
         
        '''
        #mtx: kamera matrisi ->kameranın içsel parametrelerini temsil eder. Bu matris, bir noktanın dünya koordinatlarından görüntü koordinatlarına dönüştürülmesinde kullanılır.
        # fx ,fy -> x ve y eksenlerindeki odak uzaklıklarını ifade eder. (piksel ölçek fakt.)
        # cx ,cy -> Görüntünün optik merkezini ifade eder. (x ve y eksenlerinin merkezi)
        # mtx -> kamera çözünürlüğüne, odak uzaklığına ve görüntüleme açılarına bağlı olarak değişir.

        #dist: distorsiyon katsayıları -> Gerçek dünyadaki nesnelerin optik mercekten geçerken meydana gelen bozulmaları ifade eder.
        # 2'ye ayrılır:
        #1: Radikal dist.(Küresel olmayan dist.) -> k1, k2, k3 ,p1 gibi katsayılarla ifade edilir. ( kenar bozulmalarını düzeltmek için kullanılır.)
        #Kamera lenslerinin kenarlarında veya köşelerinde görülen bir dist. türü.
        #Bu distorsiyon türü, genellikle merkezi simetrik bir yapıya sahip olmayan ve bir kenarın veya köşenin diğerlerinden farklı bir şekilde büyüdüğü veya büzüldüğü durumlarda oluşur.

        #2: Küresel dist. -> k  (lensin görüş açısını veya geniş açıyı düzelten katsayılar olarak kullanılır.)
        # Genellikle geniş açılı lensler veya fisheye lensler gibi geniş bir görüş alanına sahip lens. görülür.
        # merkezi bir noktaya yakın olan objelerin genellikle normalden daha fazla genişletilip, merkezi bir noktaya uzak olan objelerin ise daha fazla sıkıştırıldığı bir türdür.


        #rvecs: dönüş vektörü
        '''
        -Kameranın dünya koordinatlarındaki (3d) konumunu ve rotasyonunu ifade eder.
        -Euler açıları veya döndürme matrisi (rotasyon matrisi)
        -3x3 boyutuda bir matris
        -Nesnenin dünya koordinat sistemi içindeki rotasyonunu tanımlar.
        '''
        #tvecs: çevirme vektörü

        '''
        -Objenin kameraya göre konumunu temsil eder.
        - Bu vektör, nesnenin kameranın konumuna göre ötelenmesini ifade eder.
        '''
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

        # Kalibrasyon sonuçlarını kaydet (örneğin: kamera matrisi ve distorsiyon katsayıları)
        # mtx ve dist'yi ayrı dosyalarda kaydet
        #savetxt diyerek txt cinsinden kaydettik
        np.savetxt('mtx.txt', mtx)
        np.savetxt('dist.txt', dist)


        # Yeni bir kare okuyup distorsiyondan arındırın
        #Kalibre edilmiş bir resmi kaydedebilmek için distorsiyondan arındırma işlemi yapılmalıdır.
        #Bu işlemi cv2.undistort() fonk. (kameranın optik dist. giderir.) kullanarak gerçekleştiririz.


        undistorted_img = cv2.undistort(frame, mtx, dist, None)
        cv2.imwrite('kalibre_edilen_img.png', undistorted_img)
    cv2.imshow('Calibration', frame)


    key = cv2.waitKey(1)  #waitkey(delay) -> bir tuşa basılmasını bekleyeceği max süreyi milisaniye cinsinden ifade eder.
                          #eğer bu süre boyunca bir tuşa basılmazsa -1 döner.
    if key == 27:  # 'Esc' tuşuna basılınca çıkış yap. 27:esc'nin Ascii degeri
        break


#kamera kaynağını serbest bırakır.
cap.release()

#tüm pencereleri kapatır.
#cv2.destroyAllWindows()




