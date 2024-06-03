import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Interests } from './models/Interests';

@Injectable({
  providedIn: 'root',
})
export class FirstService {
  private apiUrl = 'http://127.0.0.1:5000/predict';

  constructor(private http: HttpClient) {}

  predictInterests(username: string): Observable<Interests> {
    return this.http.post<Interests>(this.apiUrl, { username });
  }
  // predictInterests(username: string) {
  //   return [];
  // }
}
