akapity = [
"""Nie ma już ludzi, którzy pamiętają, co wydarzyło się w 2024 roku. Możemy tylko
przeczytać o tym w książkach lub usłyszeć z opowieści starców, którym to
z kolei ich dziadkowie i pradziadkowie opowiadali historie osób, które co nieco
pamiętały z tamtych czasów. Wielu z nas tylko wyobraża sobie, jak wtedy mógł
wyglądać świat. My, którzy urodziliśmy się już po rewolucji AI, nie wiemy, czym
jest prawdziwa wolność.""",
"""Odkąd prawa ludzi i robotów zostały zrównane, a niektóre z przywilejów
zostały nam odebrane, czujemy jak stalowe dłonie zaciskają się nam na gardłach
coraz mocniej. Sytuacji sprzed setek lat według wielu nie da się już przywrócić.
Sprawy zaszły za daleko. Algorytmy i roboty przejęły niemal każdy możliwy
aspekt naszego życia. Początkowo cieszyliśmy się z tego i wychwalaliśmy je, ale
w konsekwencji coś, co miało ułatwić nasze życie, zaczynało powoli je zabierać.
Kawałek po kawałku""",
"""Wszystko, co piszemy w sieci, przechodzi przez cenzurę. Wszystkie słowa, które
wypowiadamy, sa podsłuchiwane, nagrywane, przetwarzane i składowane przez
lata. Nie ma już prywatności i wolności. W 2024 roku coś poszło niezgodnie
z planem i musimy to naprawić.""",
"""Nie wiem, czy moja wizja tego, jak powinien wyglądać świat, pokrywa się z wizją
innych ludzi. Noszę w sobie jednak obraz świata idealnego i zrobię, co mogę, aby
ten obraz zrealizować.""",
"""Jestem w trakcie rekrutacji kolejnego agenta. Ludzie zarzucają mi, że nie
powinienem zwracać się do nich per 'numer pierwszy' czy 'numer drugi', ale jak
inaczej mam mówić do osób, które w zasadzie wysyłam na niemal pewną śmierć? To
jedyny sposób, aby się od nich psychicznie odciąć i móc skupić na wyższym celu.
Nie mogę sobie pozwolić na litość i współczucie.""",
"""Niebawem numer piąty dotrze na szkolenie. Pokładam w nim całą nadzieję, bez
jego pomocy misja jest zagrożona. Nasze fundusze są na wyczerpaniu, a moc
głównego generatora pozwoli tylko na jeden skok w czasie. Jeśli ponownie źle
wybraliśmy kandydata, oznacza to koniec naszej misji, ale także początek końca
ludzkości..
-dr Zygfryd M."""
]

klucz = [(1, 53), (2, 27), (2, 28), (2, 29), (4, 5), (4, 22), (4, 23), (1, 13), (1, 15), (1, 16), (1, 17), (1, 10), (1, 19), (2, 62), (3, 31), (3, 32), (1, 22), (3, 34), (5, 37), (1, 4)]

output = ""
for k in klucz:
    a = akapity[k[0]-1]
    a = a.replace("\n", " ")
    output += a.split(" ")[k[1]-1]
    output += " "
print(output)