import { Component } from '@angular/core';
import { FirstService } from './first.service';
import { Observable } from 'rxjs';
import { Interests } from './models/Interests';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'pfa-interface';
  userName: string = '';
  userInterests$: Observable<Interests> | null = null;

  constructor(private service: FirstService) {}

  submitForm() {
    console.log(this.userName);
    this.userInterests$ = this.service.predictInterests(this.userName);
  }
}
