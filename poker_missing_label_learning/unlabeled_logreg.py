"""
unlabeled_logreg.py
===================
Eksik etiketli veri ile lojistik regresyon.

Sınıf: UnlabeledLogReg

İki algoritma:
  - 'label_propagation' : labeled veriden model eğit, unlabeled'ı tahmin et, birleştir
  - 'em'                : EM (Expectation-Maximization) ile iteratif tamamlama
"""

import numpy as np
from fista import FISTASelector


class UnlabeledLogReg:
	"""
	Eksik etiketli veri (X, y_obs) ile lojistik regresyon.

	y_obs = -1 olan gözlemler eksik etiketli.
	y_obs = 0 veya 1 olan gözlemler etiketli.

	Parameters
	----------
	method : str
		'label_propagation' veya 'em'
	lambdas : array-like or None
		FISTA için denenecek lambda değerleri.
	measure : str
		Lambda seçimi için kullanılacak metrik.
	max_iter_em : int
		EM algoritması için maksimum iterasyon sayısı.
	tol_em : float
		EM yakınsama toleransı.
	max_iter_fista : int
		FISTA için maksimum iterasyon sayısı.
	random_state : int or None
	"""
	
	def __init__(
			self,
			method='label_propagation',
			lambdas=None,
			measure='f1',
			max_iter_em=20,
			tol_em=1e-3,
			max_iter_fista=1000,
			random_state=42,
	):
		if method not in ('label_propagation', 'em'):
			raise ValueError(
				f"Bilinmeyen method '{method}'. "
				"'label_propagation' veya 'em' seçin."
			)
		self.method = method
		self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 1, 20)
		self.measure = measure
		self.max_iter_em = max_iter_em
		self.tol_em = tol_em
		self.max_iter_fista = max_iter_fista
		self.random_state = random_state
		self.selector = None   # eğitilmiş FISTASelector
	
	# ------------------------------------------------------------------
	# Ana fit fonksiyonu
	# ------------------------------------------------------------------
	
	def fit(self, X, y_obs, X_valid, y_valid):
		"""
		Modeli eğit.

		Parameters
		----------
		X : array (n_samples, n_features)
			Tüm eğitim feature'ları (labeled + unlabeled).
		y_obs : array (n_samples,)
			Gözlemlenen etiketler. -1 = eksik.
		X_valid : array
			Validation feature'ları (tam etiketli).
		y_valid : array
			Validation etiketleri.
		"""
		X = np.asarray(X, dtype=float)
		y_obs = np.asarray(y_obs, dtype=float)
		
		if self.method == 'label_propagation':
			y_complete = self._label_propagation(X, y_obs, X_valid, y_valid)
		else:
			y_complete = self._em(X, y_obs, X_valid, y_valid)
		
		# Tamamlanmış Y ile son FISTA eğitimi
		self.selector = FISTASelector(
			lambdas=self.lambdas,
			max_iter=self.max_iter_fista,
		)
		self.selector.fit(X, y_complete, X_valid, y_valid, measure=self.measure)
		return self
	
	# ------------------------------------------------------------------
	# Algoritma 1: Label Propagation
	# ------------------------------------------------------------------
	
	def _label_propagation(self, X, y_obs, X_valid, y_valid):
		"""
		Adım 1: Sadece etiketli gözlemlerle (S=0) bir model eğit.
		Adım 2: Bu modelle eksik etiketleri tahmin et.
		Adım 3: Tahminleri gerçek etiketlerle birleştir.

		Neden işe yarar?
		Labeled veri az olsa bile bir başlangıç modeli elde ediyoruz.
		Unlabeled veriyi bu modelle etiketleyip eğitim setini büyütüyoruz.
		"""
		labeled_mask   = (y_obs != -1)
		unlabeled_mask = (y_obs == -1)
		
		X_labeled = X[labeled_mask]
		y_labeled = y_obs[labeled_mask]
		
		# Adım 1: Labeled veriden model eğit
		init_selector = FISTASelector(
			lambdas=self.lambdas,
			max_iter=self.max_iter_fista,
		)
		init_selector.fit(X_labeled, y_labeled, X_valid, y_valid, measure=self.measure)
		
		# Adım 2: Unlabeled gözlemler için tahmin üret
		y_complete = y_obs.copy()
		if unlabeled_mask.sum() > 0:
			proba = init_selector.predict_proba(X[unlabeled_mask])
			# Olasılık 0.5'ten büyükse 1, küçükse 0
			y_complete[unlabeled_mask] = (proba >= 0.5).astype(float)
		
		return y_complete
	
	# ------------------------------------------------------------------
	# Algoritma 2: EM (Expectation-Maximization)
	# ------------------------------------------------------------------
	
	def _em(self, X, y_obs, X_valid, y_valid):
		"""
		EM algoritması — iki adımı iteratif olarak tekrarlar:

		E adımı (Expectation):
			Mevcut model ile eksik etiketlerin olasılığını hesapla.
			P(Y=1 | X, model) → soft etiket

		M adımı (Maximization):
			Bu soft etiketlerle modeli yeniden eğit.

		Neden label_propagation'dan farklı?
		Label propagation'da eksik etiketler bir kere tahmin edilip sabitleniyor.
		EM'de her iterasyonda hem etiketler hem model güncelleniyor —
		model iyileştikçe tahminler de iyileşiyor.
		"""
		labeled_mask   = (y_obs != -1)
		unlabeled_mask = (y_obs == -1)
		
		# Başlangıç: label propagation ile ilk tahmini al
		y_current = self._label_propagation(X, y_obs, X_valid, y_valid)
		
		rng = np.random.default_rng(self.random_state)
		
		for iteration in range(self.max_iter_em):
			
			# M adımı: mevcut etiketlerle modeli eğit
			selector = FISTASelector(
				lambdas=self.lambdas,
				max_iter=self.max_iter_fista,
			)
			selector.fit(X, y_current, X_valid, y_valid, measure=self.measure)
			
			# E adımı: unlabeled gözlemler için yeni olasılıkları hesapla
			if unlabeled_mask.sum() > 0:
				proba = selector.predict_proba(X[unlabeled_mask])
				y_new = y_current.copy()
				y_new[unlabeled_mask] = (proba >= 0.5).astype(float)
			else:
				break
			
			# Yakınsama kontrolü: etiketler değişmedi mi?
			n_changed = (y_new[unlabeled_mask] != y_current[unlabeled_mask]).sum()
			y_current = y_new
			
			if n_changed == 0:
				print(f"  EM {iteration+1}. iterasyonda yakınsadı.")
				break
		
		return y_current
	
	# ------------------------------------------------------------------
	# Tahmin
	# ------------------------------------------------------------------
	
	def predict_proba(self, X):
		"""Pozitif sınıf olasılığı."""
		if self.selector is None:
			raise RuntimeError("Önce fit() çağırılmalı.")
		return self.selector.predict_proba(X)
	
	def predict(self, X, threshold=0.5):
		"""Binary tahmin."""
		return (self.predict_proba(X) >= threshold).astype(int)
	
	def validate(self, X, y, measure):
		"""Validation/test metriği."""
		if self.selector is None:
			raise RuntimeError("Önce fit() çağırılmalı.")
		return self.selector.best_model.validate(X, y, measure)